import numpy as np
from scipy import stats

class FrescoAIModel:
    def __init__(self):
        pass

    def network_value(self, n_users, multiplier=100):
        return n_users * multiplier

    def ai_costs(self, n_users, infrastructure_scaling=1.2, base_cost_per_user=0.1, infrastructure_base=1000):
        variable_costs = n_users * base_cost_per_user
        infrastructure_costs = infrastructure_base * (n_users ** (infrastructure_scaling - 1))
        return variable_costs + infrastructure_costs

    def revenue_models(self, n_users, model_type, subscription_price=10, conversion_rate=0.05, 
                      usage_price=0.01, avg_usage_per_user=100, target_profit_margin=25, 
                      operational_costs_ratio=0.3):
        # Calculate base revenue
        if model_type == 'freemium':
            base_revenue = n_users * conversion_rate * subscription_price
        elif model_type == 'subscription':
            base_revenue = n_users * subscription_price
        elif model_type == 'usage_based':
            base_revenue = n_users * usage_price * avg_usage_per_user
        elif model_type == 'hybrid':
            subscription_revenue = n_users * conversion_rate * subscription_price
            usage_revenue = n_users * usage_price * avg_usage_per_user
            base_revenue = subscription_revenue + usage_revenue
        else:
            base_revenue = 0
        
        # Calculate profit-adjusted metrics
        gross_profit = base_revenue * (1 - operational_costs_ratio)
        net_profit = gross_profit * (target_profit_margin / 100)
        required_revenue = gross_profit + net_profit
        
        return {
            'base_revenue': base_revenue,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'required_revenue': required_revenue,
            'profit_margin_actual': (net_profit / base_revenue * 100) if base_revenue > 0 else 0
        }

    def analyze_market_pricing(self, gdp, population, target_percentile=0.8, target_profit_margin=25):
        # Assuming log-normal distribution for income
        mean_income = gdp / population
        sigma = 0.5  # Standard deviation parameter for log-normal distribution
        mu = np.log(mean_income) - (sigma**2) / 2

        # Calculate target income level at given percentile
        target_income = stats.lognorm.ppf(target_percentile, sigma, scale=np.exp(mu))
        
        # Calculate base price at 1% of monthly income
        base_price = target_income * 0.01 / 12
        
        # Adjust price for target profit margin
        profit_adjusted_price = base_price / (1 - target_profit_margin / 100)
        
        market_size = population * (1 - target_percentile)
        
        return {
            'recommended_price': profit_adjusted_price,
            'base_price': base_price,
            'target_income': target_income,
            'market_size': market_size,
            'profit_margin': target_profit_margin
        }

    def calculate_network_value(self, users, engagement_rate=0.7):
        self.network_value = users * engagement_rate * 100  # $100 per engaged user
        return self.network_value

    def calculate_ai_costs(self, users, compute_cost_per_user=0.5):
        self.ai_costs = users * compute_cost_per_user
        return self.ai_costs

    # This is a duplicate method, removing it as we already have a revenue_models method above