# views.py
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import google.generativeai as genai
import json
import os
from django.core.cache import cache

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY'))


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
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Read Excel file
        df = pd.read_excel(file)
        
        # Coerce numeric columns to handle strings like "N/A", "-", etc. -> NaN
        numeric_cols = [
            'flat - weighted average rate',
            'total_sales - igr',
            'total units',
            'residential_sold - igr'
            # Add more if needed, e.g., 'year'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Optional: Log bad values for debugging (remove in prod)
        bad_counts = {col: df[col].isna().sum() for col in numeric_cols if col in df.columns}
        print(f"Coerced bad values: {bad_counts}")
        
        # Flag zero-heavy locations
        zero_flags = flag_zero_locations(df)
        cache.set('zero_flags', json.dumps(zero_flags), timeout=3600)
        print(f"Zero-flagged locations: {zero_flags}")
        
        # Store data in cache (or database for production)
        data_json = df.to_json(orient='records')
        cache.set('real_estate_data', data_json, timeout=3600)
        
        # Get unique locations
        locations = df['final location'].unique().tolist()
        
        return Response({
            'message': f'Successfully loaded {len(df)} records',
            'total_records': len(df),
            'locations': locations[:20],  # First 20 locations
            'columns': df.columns.tolist(),
            'zero_flags': zero_flags  # Warn on upload
        })
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def analyze_query(request):
    """Process user query and return analysis"""
    try:
        query = request.data.get('query', '').lower()
        
        if not query:
            return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get cached data
        data_json = cache.get('real_estate_data')
        if not data_json:
            return Response({'error': 'No data loaded. Please upload a file first.'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        df = pd.read_json(data_json)
        
        # Temp debug
        amb_df = df[df['final location'] == 'Ambegaon Budruk']
        if not amb_df.empty:
            print(f"Ambegaon dtype for rate: {amb_df['flat - weighted average rate'].dtype}")
            print(f"Sample rate: {amb_df['flat - weighted average rate'].iloc[0]}")
            print(f"Year unique: {amb_df['year'].unique()}")
        
        # Extract locations from query (exact "for [loc]" + fuzzy fallback)
        locations = df['final location'].unique().tolist()
        mentioned_locations = []

        # Common words to ignore (keep for fuzzy)
        ignore_words = ['show', 'price', 'growth', 'for', 'compare', 'analyze', 'tell', 'about', 'the', 'and', 'in', 'of', 'demand', 'trend', 'trends', 'last', 'years', 'year', 'between', 'versus', 'vs']

        # Get query words (cleaned) for fuzzy
        query_words = [w.strip() for w in query.split() if len(w) > 3 and w not in ignore_words]

        # Fast path: exact match after "for" (with casing fix)
        exact_matched = False
        filtered_df = pd.DataFrame()  # Initialize to avoid undefined
        if 'for' in query:
            try:
                query_location = query.split('for')[1].strip().lower()  # Convert query location to lowercase and trim spaces
                
                # Make a copy to avoid mutating original df
                df_copy = df.copy()
                df_copy['final location'] = df_copy['final location'].str.lower().str.strip()
                
                # Check if the query location is in the DataFrame's 'final location' column (lowered)
                if query_location in df_copy['final location'].values:
                    # Get the original index from copy to find original df row
                    match_mask = df_copy['final location'] == query_location
                    copy_index = df_copy[match_mask].index[0]
                    orig_index = df.index[df.index == copy_index][0]  # Map back to original df index (assumes same order)
                    
                    # Get the original cased location
                    original_loc = df.loc[orig_index, 'final location']
                    
                    # Filter full data for this location (no risky iloc slice)
                    filtered_df = df[df['final location'] == original_loc]
                    mentioned_locations.append(original_loc)
                    exact_matched = True
                    print(f"Exact match found: {original_loc}")
                else:
                    print(f"No exact match for: {query_location}")
            except (IndexError, ValueError, KeyError) as e:
                print(f"Error parsing 'for' query: {e}")
                # Fall through to fuzzy

        # If no exact match, fallback to original fuzzy matching
        if not exact_matched:
            print("Falling back to fuzzy matching...")
            for loc in locations:
                loc_lower = loc.lower().strip()  # Trim spaces here too
                loc_words = loc_lower.split()
                
                # Check if full location is in query
                if loc_lower in query:
                    if loc not in mentioned_locations:
                        mentioned_locations.append(loc)
                else:
                    # Check if any significant word from query matches any word in location
                    for query_word in query_words:
                        for loc_word in loc_words:
                            # Match if query word is in location word (handles "ambegaon" matching "ambegaon budruk")
                            if query_word in loc_word or loc_word in query_word:
                                if loc not in mentioned_locations:
                                    mentioned_locations.append(loc)
                                break

        # Dedupe
        mentioned_locations = list(set(mentioned_locations))

        print(f"Query: {query}")
        print(f"Found locations: {mentioned_locations}")
        
        # Check if it's a general question (no specific location)
        if not mentioned_locations:
            # Check if asking for general info or list of locations
            if any(word in query for word in ['what', 'which', 'how', 'tell', 'explain', 'available', 'locations', 'areas', 'help']):
                summary = handle_general_query(query, df, locations)
                return Response({
                    'summary': summary,
                    'chart_data': [],
                    'table_data': [],
                    'total_records': 0,
                    'locations': []
                })
            else:
                return Response({
                    'summary': f"I couldn't find specific localities in your query. Available locations: {', '.join(locations[:10])}...\n\nYou can ask:\n• 'Analyze [location]'\n• 'Compare [loc1] and [loc2]'\n• 'What are the best areas?'\n• 'Show all locations'",
                    'chart_data': [],
                    'table_data': []
                })
        
        # Filter data for specific locations (unify: override if exact set it)
        if not exact_matched:
            filtered_df = df[df['final location'].isin(mentioned_locations)]
        if filtered_df.empty:
            return Response({
                'summary': f"No data found for: {', '.join(mentioned_locations)}",
                'chart_data': [],
                'table_data': []
            })
        
        # Load zero flags and check for this query
        zero_flags_json = cache.get('zero_flags', '{}')
        zero_flags = json.loads(zero_flags_json)
        query_zero_flags = {loc: zero_flags.get(loc, '') for loc in mentioned_locations if loc in zero_flags}
        warning = ""
        if any('zero' in flag.lower() for flag in query_zero_flags.values()):
            warning = f"Note: {', '.join([f'{loc}: {flag}' for loc, flag in query_zero_flags.items()])}. Data may be limited. "
        
        # Prepare chart data
        chart_data = prepare_chart_data(filtered_df, mentioned_locations)
        
        # Generate summary using Gemini
        summary = generate_gemini_summary(filtered_df, mentioned_locations, query, query_zero_flags)
        
        # Prepare table data (convert to list of dicts properly)
        table_data = []
        for idx, row in filtered_df.head(10).iterrows():
            table_data.append(row.to_dict())
        
        return Response({
            'summary': warning + summary,
            'chart_data': chart_data,
            'table_data': table_data,
            'total_records': len(filtered_df),
            'locations': mentioned_locations
        })
    
    except Exception as e:
        print(f"Error in analyze_query: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({'error': f'Server error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def handle_general_query(query, df, locations):
    """Handle general queries without specific locations"""
    try:
        # Prepare overall statistics
        total_locations = len(locations)
        avg_price = df['flat - weighted average rate'].mean()
        total_sales = df['total_sales - igr'].sum() / 10000000
        
        if 'location' in query or 'area' in query or 'available' in query or 'list' in query:
            return f"**Available Locations ({total_locations} areas)**\n\n{', '.join(locations[:20])}{'...' if len(locations) > 20 else ''}\n\nYou can ask about any of these locations!"
        
        # Use Gemini for general questions
        prompt = f"""You are a real estate expert. Answer this question based on the data context:

User Question: {query}

Data Context:
- Total locations covered: {total_locations}
- Average property price: ₹{avg_price:.0f}/sqft
- Total market sales: ₹{total_sales:.2f} Crores
- Locations: {', '.join(locations[:10])}...

Provide a helpful, concise answer (2-3 sentences). If they're asking about best areas or recommendations, mention that you can analyze specific locations if they ask."""

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        total_locations = len(locations)
        return f"I'm here to help with real estate analysis! I have data for {total_locations} locations. Try asking:\n• 'What locations are available?'\n• 'Analyze Wakad'\n• 'Compare Aundh and Akurdi'\n• 'Which area has highest prices?'"


def prepare_chart_data(df, locations):
    """Prepare data for chart visualization"""
    try:
        years = sorted(df['year'].unique())
        chart_data = []
        
        for year in years:
            year_data = {'year': int(year)}
            for loc in locations:
                loc_data = df[(df['year'] == year) & (df['final location'] == loc)]
                if not loc_data.empty:
                    # Use mean for robustness (handles NaNs/zeros/multiples)
                    avg_rate = loc_data['flat - weighted average rate'].mean()
                    if pd.notna(avg_rate) and avg_rate > 0:  # Skip pure zeros if desired
                        year_data[loc] = round(float(avg_rate))
                    else:
                        year_data[loc] = 0  # Or None for gaps
            chart_data.append(year_data)
        
        print(f"Chart data prepared: {chart_data}")
        return chart_data
    except Exception as e:
        print(f"Error in prepare_chart_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def generate_gemini_summary(df, locations, query, query_zero_flags):
    """Generate summary using Gemini AI"""
    try:
        # Prepare data summary for Gemini
        data_summary = []
        for loc in locations:
            loc_data = df[df['final location'] == loc]
            if loc_data.empty:
                continue
                
            latest_year = loc_data['year'].max()
            latest_data = loc_data[loc_data['year'] == latest_year].iloc[0]
            
            data_summary.append({
                'location': loc,
                'year': int(latest_year),
                'avg_price': float(latest_data['flat - weighted average rate']) if pd.notna(latest_data['flat - weighted average rate']) else 0,
                'total_sales': float(latest_data['total_sales - igr']) if pd.notna(latest_data['total_sales - igr']) else 0,
                'total_units': float(latest_data['total units']) if pd.notna(latest_data['total units']) else 0,
                'residential_sold': float(latest_data['residential_sold - igr']) if pd.notna(latest_data['residential_sold - igr']) else 0
            })
        
        print(f"Data summary for Gemini: {data_summary}")
        
        # Create prompt for Gemini
        prompt = f"""You are a real estate analyst. Analyze the following data and provide a concise summary (3-4 sentences) based on the user's query.

User Query: {query}

Data:
{json.dumps(data_summary, indent=2)}

Additional Context: {json.dumps(query_zero_flags)}  # E.g., {'Ambegaon Budruk': '75.0% zero/missing'}

Provide insights about:
- Price trends (note if 0/missing indicates low activity)
- Sales volume (0 units = no transactions)
- Market demand
- Comparison if multiple locations

If data is mostly 0 or missing, say 'Limited recorded activity—consider recent market updates.' Keep it concise, professional, and actionable. Use Indian Rupee format (₹ and Crores)."""

        # Call Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        print(f"Gemini response: {response.text}")
        return response.text
    
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to basic summary if Gemini fails
        return generate_fallback_summary(df, locations, query)


def generate_fallback_summary(df, locations, query):
    """Fallback summary if Gemini API fails"""
    try:
        loc = locations[0]
        loc_data = df[df['final location'] == loc]
        
        if loc_data.empty:
            return f"No data available for {loc}"
            
        latest_year = loc_data['year'].max()
        latest_data = loc_data[loc_data['year'] == latest_year].iloc[0]
        
        avg_price = pd.to_numeric(latest_data['flat - weighted average rate'], errors='coerce')
        if pd.notna(avg_price) and avg_price > 0:
            avg_price = round(avg_price)
            price_note = ""
        else:
            avg_price = 0
            price_note = " (no price data available)"
        
        total_sales_val = pd.to_numeric(latest_data['total_sales - igr'], errors='coerce')
        total_sales = (total_sales_val / 10000000) if pd.notna(total_sales_val) else 0
        
        total_units = pd.to_numeric(latest_data['total units'], errors='coerce')
        if pd.notna(total_units) and total_units > 0:
            total_units = round(total_units)
            units_note = ""
        else:
            total_units = 0
            units_note = " (no units sold)"
        
        if 'compare' in query and len(locations) > 1:
            summary = f"Comparison for {' vs '.join(locations)}:\n"
            latest_year = df['year'].max()  # Assume same year
            for l in locations:
                l_data = df[(df['final location'] == l) & (df['year'] == latest_year)]
                if not l_data.empty:
                    l_row = l_data.iloc[0]
                    l_price = pd.to_numeric(l_row['flat - weighted average rate'], errors='coerce')
                    l_price = round(l_price) if pd.notna(l_price) and l_price > 0 else 0
                    l_units = pd.to_numeric(l_row['total units'], errors='coerce')
                    l_units = round(l_units) if pd.notna(l_units) and l_units > 0 else 0
                    summary += f"{l}: ₹{l_price}/sqft, {l_units} units\n"
            return summary
        
        if total_units == 0:
            return f"Analysis for {loc}: No units sold in {int(latest_year)}{price_note}. Market shows no activity this year—consider recent updates."
        else:
            return f"Analysis for {loc}: Average flat price is ₹{avg_price}/sqft{price_note} ({int(latest_year)}). Total sales: ₹{total_sales:.2f} Cr with {total_units} units sold{units_note}. Market shows steady activity in this locality."
    
    except Exception as e:
        print(f"Error in fallback summary: {str(e)}")
        return f"Analysis for {locations[0]}: Data available but summary generation failed."


@api_view(['POST'])
def download_data(request):
    """Return filtered data for download"""
    try:
        locations = request.data.get('locations', [])
        
        data_json = cache.get('real_estate_data')
        if not data_json:
            return Response({'error': 'No data available'}, status=status.HTTP_400_BAD_REQUEST)
        
        df = pd.read_json(data_json)
        
        if locations:
            df = df[df['final location'].isin(locations)]
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        return Response({
            'csv_data': csv_data,
            'filename': 'real_estate_filtered.csv'
        })
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)