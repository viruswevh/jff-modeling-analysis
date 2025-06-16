from flask import Flask, request, jsonify
from flask_cors import CORS
from fresco_ai_model import FrescoAIModel
import numpy as np

app = Flask(__name__)
CORS(app)
model = FrescoAIModel()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    analysis_type = data['analysisType']
    params = data['params']

    if analysis_type == 'cost_structure':
        user_counts = np.logspace(2, 6, 50)  # 100 to 1M users
        scenarios = {
            'Conservative (1.1x)': [model.ai_costs(n, infrastructure_scaling=1.1) for n in user_counts],
            'Moderate (1.2x)': [model.ai_costs(n, infrastructure_scaling=1.2) for n in user_counts],
            'Aggressive (1.3x)': [model.ai_costs(n, infrastructure_scaling=1.3) for n in user_counts]
        }
        return jsonify({
            'title': 'Cost Structure Analysis',
            'graphData': {
                'labels': user_counts.tolist(),
                'datasets': [{
                    'label': label,
                    'data': costs,
                    'fill': False,
                    'borderColor': f'hsl({i * 360/len(scenarios)}, 70%, 50%)',
                } for i, (label, costs) in enumerate(scenarios.items())]
            }
        })

    elif analysis_type == 'network_value':
        users = np.linspace(100, params['n_users'], 50)
        values = [model.network_value(n, params['network_value_multiplier']) for n in users]
        return jsonify({
            'title': 'Network Value Growth',
            'graphData': {
                'labels': users.tolist(),
                'datasets': [{
                    'label': 'Network Value',
                    'data': values,
                    'borderColor': 'rgb(75, 192, 192)',
                    'fill': False
                }]
            }
        })

    elif analysis_type == 'user_growth':
        time_points = np.arange(1, params['months'] + 1)
        scenarios = {
            'Linear Growth': np.linspace(100, params['max_users'], params['months']),
            'Exponential Growth': 100 * np.exp(np.linspace(0, np.log(params['max_users']/100), params['months'])),
            'S-Curve Growth': params['max_users'] / (1 + np.exp(-0.3 * (time_points - params['months']/2))),
            'Rapid Early Growth': params['max_users'] * (1 - np.exp(-0.15 * time_points))
        }
        return jsonify({
            'title': 'User Growth Scenarios',
            'graphData': {
                'labels': time_points.tolist(),
                'datasets': [{
                    'label': label,
                    'data': data.tolist(),
                    'fill': False,
                    'borderColor': f'hsl({i * 360/len(scenarios)}, 70%, 50%)',
                } for i, (label, data) in enumerate(scenarios.items())]
            }
        })

    elif analysis_type == 'revenue_models':
        user_counts = np.linspace(100, params['n_users'], 50)
        models = ['freemium', 'subscription', 'usage_based', 'hybrid']
        revenues = {}
        
        for model_name in models:
            model_data = []
            for n in user_counts:
                result = model.revenue_models(n, model_name, 
                                           subscription_price=params['subscription_price'],
                                           conversion_rate=params['conversion_rate'],
                                           usage_price=params['usage_price'],
                                           avg_usage_per_user=params['avg_usage_per_user'],
                                           target_profit_margin=params.get('target_profit_margin', 25),
                                           operational_costs_ratio=params.get('operational_costs_ratio', 0.3))
                model_data.append(result['required_revenue'])  # Use profit-adjusted revenue
            revenues[model_name] = model_data
        
        return jsonify({
            'title': 'Revenue Models Comparison (Profit-Adjusted)',
            'graphData': {
                'labels': user_counts.tolist(),
                'datasets': [{
                    'label': f'{label.title()} (Target: {params.get("target_profit_margin", 25)}% margin)',
                    'data': data,
                    'fill': False,
                    'borderColor': f'hsl({i * 360/len(models)}, 70%, 50%)',
                } for i, (label, data) in enumerate(revenues.items())]
            }
        })

    elif analysis_type == 'market_pricing':
        target_percentiles = np.linspace(0.1, 0.9, 50)  # 10th to 90th percentile
        
        prices = []
        market_sizes = []
        for percentile in target_percentiles:
            result = model.analyze_market_pricing(
                gdp=params['gdp'],
                population=params['population'],
                target_percentile=percentile,
                target_profit_margin=params.get('target_profit_margin', 25)
            )
            prices.append(result['recommended_price'])
            market_sizes.append(result['market_size'])
        
        return jsonify({
            'title': f'Market Pricing Analysis (Target: {params.get("target_profit_margin", 25)}% Profit Margin)',
            'graphData': {
                'labels': [f'{p:.1f}%' for p in (target_percentiles * 100)],
                'datasets': [
                    {
                        'label': 'Profit-Adjusted Price ($)',
                        'data': prices,
                        'fill': False,
                        'borderColor': 'rgb(75, 192, 192)',
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'Market Size (Users)',
                        'data': market_sizes,
                        'fill': False,
                        'borderColor': 'rgb(255, 99, 132)',
                        'yAxisID': 'y1'
                    }
                ]
            }
        })

    return jsonify({'error': 'Invalid analysis type'})

if __name__ == '__main__':
    app.run(debug=True)