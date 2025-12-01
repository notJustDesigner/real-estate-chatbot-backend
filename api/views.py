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
        
        # Store data in cache (or database for production)
        data_json = df.to_json(orient='records')
        cache.set('real_estate_data', data_json, timeout=3600)
        
        # Get unique locations
        locations = df['final location'].unique().tolist()
        
        return Response({
            'message': f'Successfully loaded {len(df)} records',
            'total_records': len(df),
            'locations': locations[:20],  # First 20 locations
            'columns': df.columns.tolist()
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
        
        # Extract locations from query (improved fuzzy matching)
        locations = df['final location'].unique()
        mentioned_locations = []
        
        # Common words to ignore
        ignore_words = ['show', 'price', 'growth', 'for', 'compare', 'analyze', 'tell', 'about', 'the', 'and', 'in', 'of', 'demand', 'trend', 'trends', 'last', 'years', 'year', 'between', 'versus', 'vs']
        
        # Get query words (cleaned)
        query_words = [w.strip() for w in query.split() if len(w) > 3 and w not in ignore_words]
        
        for loc in locations:
            loc_lower = loc.lower()
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
        
        # Filter data for specific locations
        filtered_df = df[df['final location'].isin(mentioned_locations)]
        
        if filtered_df.empty:
            return Response({
                'summary': f"No data found for: {', '.join(mentioned_locations)}",
                'chart_data': [],
                'table_data': []
            })
        
        # Prepare chart data
        chart_data = prepare_chart_data(filtered_df, mentioned_locations)
        
        # Generate summary using Gemini
        summary = generate_gemini_summary(filtered_df, mentioned_locations, query)
        
        # Prepare table data (convert to list of dicts properly)
        table_data = []
        for idx, row in filtered_df.head(10).iterrows():
            table_data.append(row.to_dict())
        
        return Response({
            'summary': summary,
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
                    # Handle NaN values
                    avg_rate = loc_data['flat - weighted average rate'].iloc[0]
                    if pd.notna(avg_rate):
                        year_data[loc] = round(float(avg_rate))
            chart_data.append(year_data)
        
        print(f"Chart data prepared: {chart_data}")
        return chart_data
    except Exception as e:
        print(f"Error in prepare_chart_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def generate_gemini_summary(df, locations, query):
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

Provide insights about:
- Price trends
- Sales volume
- Market demand
- Comparison if multiple locations

Keep it concise, professional, and actionable. Use Indian Rupee format (₹ and Crores)."""

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
        
        avg_price = round(latest_data['flat - weighted average rate']) if pd.notna(latest_data['flat - weighted average rate']) else 0
        total_sales = (latest_data['total_sales - igr'] / 10000000) if pd.notna(latest_data['total_sales - igr']) else 0
        total_units = round(latest_data['total units']) if pd.notna(latest_data['total units']) else 0
        
        if 'compare' in query and len(locations) > 1:
            summary = f"Comparison for {' vs '.join(locations)}:\n"
            for l in locations:
                l_data = df[(df['final location'] == l) & (df['year'] == latest_year)]
                if not l_data.empty:
                    l_row = l_data.iloc[0]
                    l_price = round(l_row['flat - weighted average rate']) if pd.notna(l_row['flat - weighted average rate']) else 0
                    l_units = round(l_row['total units']) if pd.notna(l_row['total units']) else 0
                    summary += f"{l}: ₹{l_price}/sqft, {l_units} units\n"
            return summary
        
        return f"Analysis for {loc}: Average flat price is ₹{avg_price}/sqft ({int(latest_year)}). Total sales: ₹{total_sales:.2f} Cr with {total_units} units sold. Market shows steady activity in this locality."
    
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