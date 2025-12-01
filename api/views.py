# views.py
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import google.generativeai as genai
import json
import os
import math
from django.core.cache import cache
from io import StringIO  # Fix for pd.read_json deprecation


# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY'))


def sanitize(obj):
    """Recursively replace NaN & Inf with 0 for JSON compliance"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj


def flag_zero_locations(df, threshold=0.5):
    """Flag locations with high zero/NaN rates in numeric columns"""
    numeric_cols = [
        'flat - weighted average rate',
        'total_sales - igr',
        'total units',
        'residential_sold - igr'
    ]
    zero_flags = {}
    for loc in df['final location'].unique():
        loc_df = df[df['final location'] == loc]
        zero_count = 0
        total_possible = 0
        for col in numeric_cols:
            if col in loc_df.columns:
                total_possible += 1
                zeros_nans = loc_df[col].isna().sum() + (loc_df[col] == 0).sum()
                zero_count += zeros_nans
        zero_ratio = zero_count / total_possible if total_possible > 0 else 1
        if zero_ratio > threshold:
            zero_flags[loc] = f"{zero_ratio:.1%} of metrics are zero/missing"
    return zero_flags


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_file(request):
    """Upload and parse Excel file"""
    try:
        file = request.FILES.get('file')
        if not file:
            return Response(sanitize({'error': 'No file provided'}), status=status.HTTP_400_BAD_REQUEST)

        df = pd.read_excel(file)

        # Convert bad numeric values → NaN
        numeric_cols = [
            'flat - weighted average rate',
            'total_sales - igr',
            'total units',
            'residential_sold - igr',
            'year'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.fillna(0)  # Extra safety: NaN → 0

        zero_flags = flag_zero_locations(df)
        cache.set('zero_flags', json.dumps(zero_flags), timeout=3600)

        data_json = df.to_json(orient='records')
        cache.set('real_estate_data', data_json, timeout=3600)

        locations = df['final location'].unique().tolist()

        return Response(sanitize({
            'message': f'Successfully loaded {len(df)} records',
            'total_records': len(df),
            'locations': locations[:20],
            'columns': df.columns.tolist(),
            'zero_flags': zero_flags
        }), status=status.HTTP_200_OK)

    except Exception as e:
        return Response(sanitize({'error': str(e)}), status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['POST'])
def analyze_query(request):
    """Process user query and return analysis"""
    try:
        query = request.data.get('query', '').lower().strip()

        if not query:
            return Response(sanitize({'error': 'No query provided'}), status=status.HTTP_400_BAD_REQUEST)

        data_json = cache.get('real_estate_data')
        if not data_json:
            return Response(sanitize({'error': 'No data loaded. Please upload a file first.'}), status=status.HTTP_400_BAD_REQUEST)

        df = pd.read_json(StringIO(data_json))

        # Coerce again for safety + sanitize NaN
        numeric_cols = ['flat - weighted average rate', 'total_sales - igr', 'total units', 'residential_sold - igr', 'year']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.fillna(0)

        locations = df['final location'].unique().tolist()
        mentioned_locations = []
        for loc in locations:
            loc_lower = loc.lower().strip()
            if loc_lower in query:
                mentioned_locations.append(loc)

        mentioned_locations = list(set(mentioned_locations))

        if not mentioned_locations:
            return Response(sanitize({
                'summary': f"No specific locations detected. First 10 available: {', '.join(locations[:10])}...",
                'chart_data': [],
                'table_data': []
            }), status=status.HTTP_200_OK)

        filtered_df = df[df['final location'].isin(mentioned_locations)]
        if filtered_df.empty:
            return Response(sanitize({
                'summary': f"No data found for: {', '.join(mentioned_locations)}",
                'chart_data': [],
                'table_data': []
            }), status=status.HTTP_200_OK)

        # Prepare chart data
        chart_data = prepare_chart_data(filtered_df, mentioned_locations)

        # Generate summary from Gemini
        summary = generate_gemini_summary(filtered_df, mentioned_locations, query, {})

        table_data = []
        for _, row in filtered_df.head(10).iterrows():
            table_data.append(row.to_dict())

        response_data = {
            'summary': summary,
            'chart_data': chart_data,
            'table_data': table_data,
            'total_records': len(filtered_df),
            'locations': mentioned_locations
        }

        return Response(sanitize(response_data), status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Error in analyze_query:", e)
        import traceback
        traceback.print_exc()
        return Response(sanitize({'error': str(e)}), status=status.HTTP_500_INTERNAL_SERVER_ERROR)



def prepare_chart_data(df, locations):
    """Prepare data for chart visualization"""
    try:
        years = sorted(df['year'].unique())
        chart_data = []
        for year in years:
            year_entry = {'year': int(year)}
            for loc in locations:
                loc_data = df[(df['year'] == year) & (df['final location'] == loc)]
                year_entry[loc] = round(loc_data['flat - weighted average rate'].mean()) if not loc_data.empty else 0
            chart_data.append(year_entry)
        return chart_data
    except:
        return []



def generate_gemini_summary(df, locations, query, query_zero_flags):
    """Generate summary using Gemini AI"""
    try:
        data_summary = []
        for loc in locations:
            loc_df = df[df['final location'] == loc]
            if loc_df.empty:
                continue
            latest_year = loc_df['year'].max()
            latest_row = loc_df[loc_df['year'] == latest_year].iloc[0]
            data_summary.append({
                "location": loc,
                "year": int(latest_year),
                "avg_price": float(latest_row.get('flat - weighted average rate', 0)),
                "total_sales": float(latest_row.get('total_sales - igr', 0))
            })

        prompt = f"""
        Analyze real estate data for: {', '.join(locations)}.
        User Query: {query}

        Data:
        {json.dumps(data_summary, indent=2)}

        If values are 0 → treat as missing data.
        Keep summary concise (3-4 lines), professional, Indian pricing format.
        """

        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content(prompt)
        return res.text

    except Exception as e:
        print("Gemini error:", e)
        return f"Limited recorded activity in {', '.join(locations)} — consider recent market updates."



@api_view(['POST'])
def download_data(request):
    """Return filtered data for download"""
    try:
        locations = request.data.get('locations', [])

        data_json = cache.get('real_estate_data')
        if not data_json:
            return Response(sanitize({'error': 'No data found'}), status=status.HTTP_400_BAD_REQUEST)

        df = pd.read_json(StringIO(data_json))

        numeric_cols = ['flat - weighted average rate', 'total_sales - igr', 'total units', 'year']
        for col in numeric_cols:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.fillna(0)

        if locations:
            df = df[df['final location'].isin(locations)]

        csv_data = df.to_csv(index=False)

        return Response(sanitize({
            'csv_data': csv_data,
            'filename': 'real_estate_filtered.csv'
        }), status=status.HTTP_200_OK)

    except Exception as e:
        return Response(sanitize({'error': str(e)}), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
