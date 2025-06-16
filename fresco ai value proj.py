import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import seaborn as sns
from scipy.optimize import minimize_scalar
from scipy import stats  # Add this missing import
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FrescoAIModel:
    def __init__(self):
        """Initialize the Fresco AI cost and network value model"""
        self.base_cost_per_user = 0.1  # Base AI processing cost per user per month
        self.infrastructure_base = 1000  # Base infrastructure cost
        self.network_value_multiplier = 100  # Multiplier for network value calculation
        
    def network_value(self, n_users, multiplier=None):
        """
        Calculate network value using n*log(n) formula
        Args:
            n_users: Number of users
            multiplier: Value multiplier (default uses instance variable)
        Returns:
            Network value
        """
        if multiplier is None:
            multiplier = self.network_value_multiplier
        
        if n_users <= 1:
            return 0
        return multiplier * n_users * np.log(n_users)
    
    def ai_costs(self, n_users, cost_per_user=None, infrastructure_scaling=1.2):
        """
        Calculate AI running costs
        Args:
            n_users: Number of users
            cost_per_user: Cost per user (default uses instance variable)
            infrastructure_scaling: How infrastructure costs scale with users
        Returns:
            Total monthly AI costs
        """
        if cost_per_user is None:
            cost_per_user = self.base_cost_per_user
            
        # Linear cost per user + infrastructure scaling
        user_costs = n_users * cost_per_user
        infrastructure_costs = self.infrastructure_base * (n_users ** (infrastructure_scaling - 1))
        
        return user_costs + infrastructure_costs
    
    def analyze_market_pricing(self, gdp, population, target_percentile=50, 
                             price_to_income_ratio=0.01, shape_parameter=1.0):
        """Calculate optimal pricing based on income distribution
        Args:
            gdp: Country's GDP in USD
            population: Country's population
            target_percentile: Target market percentile (0-100)
            price_to_income_ratio: Expected price as ratio of monthly income
            shape_parameter: Controls income distribution skewness
        Returns:
            dict with pricing analysis results
        """
        # Calculate mean income
        mean_income = gdp / population
        
        # Create lognormal distribution model
        scale = mean_income / np.exp(shape_parameter ** 2 / 2)  # Adjust for lognormal mean
        income_dist = stats.lognorm(s=shape_parameter, scale=scale)
        
        # Calculate target income level and price
        target_income = income_dist.ppf(target_percentile/100)
        recommended_price = target_income * price_to_income_ratio
        
        # Calculate market size and potential revenue
        market_size = population * (1 - target_percentile/100)
        potential_revenue = market_size * recommended_price
        
        # Calculate price elasticity estimate
        elasticity = -1.5 - (target_percentile/100)  # Simple model: higher percentile = more elastic
        
        return {
            'recommended_price': recommended_price,
            'target_income_level': target_income,
            'market_size': market_size,
            'potential_revenue': potential_revenue,
            'price_elasticity': elasticity,
            'mean_income': mean_income
        }
    
    def optimize_pricing_strategy(self, gdp, population, min_percentile=10, 
                                max_percentile=90, steps=20):
        """Find optimal pricing strategy by analyzing different market segments
        Args:
            gdp: Country's GDP in USD
            population: Country's population
            min_percentile: Minimum market percentile to consider
            max_percentile: Maximum market percentile to consider
            steps: Number of analysis points
        Returns:
            dict with optimal pricing strategy
        """
        percentiles = np.linspace(min_percentile, max_percentile, steps)
        results = []
        
        for p in percentiles:
            analysis = self.analyze_market_pricing(gdp, population, target_percentile=p)
            revenue = analysis['potential_revenue']
            costs = self.ai_costs(analysis['market_size'])
            profit = revenue - costs
            
            results.append({
                'percentile': p,
                'price': analysis['recommended_price'],
                'market_size': analysis['market_size'],
                'revenue': revenue,
                'costs': costs,
                'profit': profit
            })
        
        # Find optimal strategy
        optimal = max(results, key=lambda x: x['profit'])
        
        return {
            'optimal_strategy': optimal,
            'all_strategies': results
        }

    def revenue_models(self, n_users, pricing_model='freemium', 
                      subscription_price=10, conversion_rate=0.05, 
                      usage_price=0.01, avg_usage_per_user=100,
                      income_based_pricing=None):
        """Calculate revenue based on different pricing models
        Args:
            n_users: Number of users
            pricing_model: 'freemium', 'subscription', 'usage_based', 'hybrid', 'income_based'
            subscription_price: Monthly subscription price
            conversion_rate: Free to paid conversion rate
            usage_price: Price per usage unit
            avg_usage_per_user: Average usage units per user
            income_based_pricing: Dict with gdp and population for income-based pricing
        Returns:
            Monthly revenue
        """
        if pricing_model == 'income_based' and income_based_pricing:
            analysis = self.analyze_market_pricing(
                income_based_pricing['gdp'],
                income_based_pricing['population']
            )
            return analysis['market_size'] * analysis['recommended_price']
            
        if pricing_model == 'freemium':
            paying_users = n_users * conversion_rate
            return paying_users * subscription_price
        
        elif pricing_model == 'subscription':
            return n_users * subscription_price
        
        elif pricing_model == 'usage_based':
            return n_users * usage_price * avg_usage_per_user
        
        elif pricing_model == 'hybrid':
            # 50% subscription, 50% usage-based
            sub_revenue = n_users * conversion_rate * subscription_price
            usage_revenue = n_users * usage_price * avg_usage_per_user * 0.5
            return sub_revenue + usage_revenue
        
        return 0

# Initialize the model
model = FrescoAIModel()

# Create user growth scenarios
def create_user_scenarios(max_users=100000, months=36):
    """Create different user growth scenarios"""
    time_points = np.arange(1, months + 1)
    
    scenarios = {
        'Linear Growth': np.linspace(100, max_users, months),
        'Exponential Growth': 100 * np.exp(np.linspace(0, np.log(max_users/100), months)),
        'S-Curve Growth': max_users / (1 + np.exp(-0.3 * (time_points - months/2))),
        'Rapid Early Growth': max_users * (1 - np.exp(-0.15 * time_points))
    }
    
    return time_points, scenarios

# Analysis functions
def analyze_cost_structure():
    """Analyze how costs scale with user growth"""
    user_counts = np.logspace(2, 6, 50)  # 100 to 1M users
    
    # Different cost scaling scenarios
    scenarios = {
        'Conservative (1.1x)': [model.ai_costs(n, infrastructure_scaling=1.1) for n in user_counts],
        'Moderate (1.2x)': [model.ai_costs(n, infrastructure_scaling=1.2) for n in user_counts],
        'Aggressive (1.5x)': [model.ai_costs(n, infrastructure_scaling=1.5) for n in user_counts]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot absolute costs
    for scenario, costs in scenarios.items():
        ax1.loglog(user_counts, costs, label=scenario, linewidth=2)
    
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Monthly AI Costs ($)')
    ax1.set_title('AI Cost Scaling with User Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cost per user
    for scenario, costs in scenarios.items():
        cost_per_user = np.array(costs) / user_counts
        ax2.semilogx(user_counts, cost_per_user, label=scenario, linewidth=2)
    
    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Cost per User ($)')
    ax2.set_title('Cost per User vs Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_network_value():
    """Analyze network value growth patterns"""
    user_counts = np.logspace(2, 6, 50)
    
    # Different network value multipliers
    multipliers = [50, 100, 200, 500]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for mult in multipliers:
        values = [model.network_value(n, mult) for n in user_counts]
        ax1.loglog(user_counts, values, label=f'Multiplier: {mult}', linewidth=2)
    
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Network Value ($)')
    ax1.set_title('Network Value Growth (n*log(n))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Compare with other network effects
    linear_effect = user_counts * 100
    quadratic_effect = user_counts ** 2 * 0.01
    nlogn_effect = [model.network_value(n, 100) for n in user_counts]
    
    ax2.loglog(user_counts, linear_effect, label='Linear (n)', linewidth=2)
    ax2.loglog(user_counts, nlogn_effect, label='n*log(n)', linewidth=2)
    ax2.loglog(user_counts, quadratic_effect, label='Quadratic (n²)', linewidth=2)
    
    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Relative Network Value')
    ax2.set_title('Network Effect Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_pricing_models():
    """Compare different pricing model effectiveness"""
    time_points, user_scenarios = create_user_scenarios(50000, 24)
    
    pricing_models = ['freemium', 'subscription', 'usage_based', 'hybrid']
    colors = ['blue', 'green', 'red', 'orange']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (model_name, users) in enumerate(user_scenarios.items()):
        ax = axes[i]
        
        for j, pricing_model in enumerate(pricing_models):
            revenues = [model.revenue_models(n, pricing_model) for n in users]
            costs = [model.ai_costs(n) for n in users]
            profits = np.array(revenues) - np.array(costs)
            
            ax.plot(time_points, profits, label=f'{pricing_model.replace("_", " ").title()}', 
                   color=colors[j], linewidth=2)
        
        ax.set_title(f'Profit Analysis: {model_name}')
        ax.set_xlabel('Months')
        ax.set_ylabel('Monthly Profit ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def interactive_parameter_analysis():
    """Create interactive analysis with parameter sliders"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial parameters
    max_users = 50000
    months = 24
    cost_per_user = 0.1
    conversion_rate = 0.05
    subscription_price = 10
    
    time_points = np.arange(1, months + 1)
    users = np.linspace(1000, max_users, months)
    
    def update_plots(cost_per_user, conversion_rate, subscription_price, network_multiplier):
        # Clear axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Calculate metrics
        costs = [model.ai_costs(n, cost_per_user) for n in users]
        revenues = [model.revenue_models(n, 'freemium', subscription_price, conversion_rate) for n in users]
        profits = np.array(revenues) - np.array(costs)
        network_values = [model.network_value(n, network_multiplier) for n in users]
        
        # Plot 1: Cost vs Revenue
        ax1.plot(time_points, costs, label='Costs', color='red', linewidth=2)
        ax1.plot(time_points, revenues, label='Revenue', color='green', linewidth=2)
        ax1.set_title('Costs vs Revenue Over Time')
        ax1.set_xlabel('Months')
        ax1.set_ylabel('Amount ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Profit
        ax2.plot(time_points, profits, color='blue', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Monthly Profit')
        ax2.set_xlabel('Months')
        ax2.set_ylabel('Profit ($)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Network Value
        ax3.plot(users, network_values, color='purple', linewidth=2)
        ax3.set_title('Network Value Growth')
        ax3.set_xlabel('Number of Users')
        ax3.set_ylabel('Network Value ($)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Unit Economics
        cost_per_user_actual = np.array(costs) / users
        revenue_per_user = np.array(revenues) / users
        
        ax4.plot(time_points, cost_per_user_actual, label='Cost per User', color='red', linewidth=2)
        ax4.plot(time_points, revenue_per_user, label='Revenue per User', color='green', linewidth=2)
        ax4.set_title('Unit Economics')
        ax4.set_xlabel('Months')
        ax4.set_ylabel('Amount per User ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.draw()
    
    # Initial plot
    update_plots(cost_per_user, conversion_rate, subscription_price, 100)
    
    # Create sliders
    ax_cost = plt.axes([0.2, 0.15, 0.5, 0.03])
    ax_conversion = plt.axes([0.2, 0.10, 0.5, 0.03])
    ax_price = plt.axes([0.2, 0.05, 0.5, 0.03])
    ax_network = plt.axes([0.2, 0.00, 0.5, 0.03])
    
    slider_cost = Slider(ax_cost, 'Cost per User', 0.01, 1.0, valinit=cost_per_user)
    slider_conversion = Slider(ax_conversion, 'Conversion Rate', 0.01, 0.2, valinit=conversion_rate)
    slider_price = Slider(ax_price, 'Subscription Price', 5, 50, valinit=subscription_price)
    slider_network = Slider(ax_network, 'Network Multiplier', 10, 500, valinit=100)
    
    def update(val):
        update_plots(slider_cost.val, slider_conversion.val, slider_price.val, slider_network.val)
    
    slider_cost.on_changed(update)
    slider_conversion.on_changed(update)
    slider_price.on_changed(update)
    slider_network.on_changed(update)
    
    plt.show()

def break_even_analysis():
    """Analyze break-even points for different scenarios"""
    user_ranges = np.logspace(2, 5, 100)  # 100 to 100k users
    
    pricing_scenarios = {
        'Conservative': {'price': 5, 'conversion': 0.02},
        'Moderate': {'price': 10, 'conversion': 0.05},
        'Aggressive': {'price': 20, 'conversion': 0.08}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'green', 'red']
    
    for i, (scenario, params) in enumerate(pricing_scenarios.items()):
        costs = [model.ai_costs(n) for n in user_ranges]
        revenues = [model.revenue_models(n, 'freemium', 
                                       params['price'], params['conversion']) for n in user_ranges]
        profits = np.array(revenues) - np.array(costs)
        
        # Find break-even point
        break_even_idx = np.where(profits > 0)[0]
        if len(break_even_idx) > 0:
            break_even_users = user_ranges[break_even_idx[0]]
            ax1.axvline(x=break_even_users, color=colors[i], linestyle='--', alpha=0.7,
                       label=f'{scenario} Break-even: {break_even_users:.0f} users')
        
        ax1.semilogx(user_ranges, profits, color=colors[i], linewidth=2, label=scenario)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Monthly Profit ($)')
    ax1.set_title('Break-even Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sensitivity analysis
    conversion_rates = np.linspace(0.01, 0.15, 20)
    subscription_prices = np.linspace(5, 25, 20)
    
    break_even_matrix = np.zeros((len(conversion_rates), len(subscription_prices)))
    
    for i, conv_rate in enumerate(conversion_rates):
        for j, price in enumerate(subscription_prices):
            # Find break-even for 10k users scenario
            cost_10k = model.ai_costs(10000)
            revenue_10k = model.revenue_models(10000, 'freemium', price, conv_rate)
            profit_10k = revenue_10k - cost_10k
            break_even_matrix[i, j] = profit_10k
    
    im = ax2.imshow(break_even_matrix, cmap='RdYlGn', aspect='auto', origin='lower')
    ax2.set_xlabel('Subscription Price ($)')
    ax2.set_ylabel('Conversion Rate')
    ax2.set_title('Profit Heatmap (10K Users)')
    
    # Set ticks
    ax2.set_xticks(np.arange(0, len(subscription_prices), 5))
    ax2.set_xticklabels([f'{subscription_prices[i]:.0f}' for i in range(0, len(subscription_prices), 5)])
    ax2.set_yticks(np.arange(0, len(conversion_rates), 5))
    ax2.set_yticklabels([f'{conversion_rates[i]:.2f}' for i in range(0, len(conversion_rates), 5)])
    
    plt.colorbar(im, ax=ax2, label='Monthly Profit ($)')
    plt.tight_layout()
    plt.show()

def scenario_comparison():
    """Compare different growth and pricing scenarios"""
    time_points, user_scenarios = create_user_scenarios(100000, 36)
    
    # Create comprehensive comparison
    results = []
    
    pricing_models = ['freemium', 'subscription', 'usage_based', 'hybrid']
    
    for scenario_name, users in user_scenarios.items():
        for pricing_model in pricing_models:
            costs = [model.ai_costs(n) for n in users]
            revenues = [model.revenue_models(n, pricing_model) for n in users]
            profits = np.array(revenues) - np.array(costs)
            network_values = [model.network_value(n) for n in users]
            
            # Calculate key metrics
            final_profit = profits[-1]
            months_to_profitability = np.where(profits > 0)[0]
            months_to_profit = months_to_profitability[0] + 1 if len(months_to_profitability) > 0 else None
            total_profit = np.sum(profits[profits > 0]) if len(profits[profits > 0]) > 0 else 0
            final_network_value = network_values[-1]
            
            results.append({
                'Growth Scenario': scenario_name,
                'Pricing Model': pricing_model,
                'Final Monthly Profit': final_profit,
                'Months to Profitability': months_to_profit,
                'Total 3-Year Profit': total_profit,
                'Final Network Value': final_network_value,
                'Final Users': users[-1]
            })
    
    df_results = pd.DataFrame(results)
    
    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Final Monthly Profit by scenario
    pivot1 = df_results.pivot(index='Growth Scenario', columns='Pricing Model', values='Final Monthly Profit')
    pivot1.plot(kind='bar', ax=ax1)
    ax1.set_title('Final Monthly Profit by Scenario')
    ax1.set_ylabel('Monthly Profit ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Months to Profitability
    pivot2 = df_results.pivot(index='Growth Scenario', columns='Pricing Model', values='Months to Profitability')
    pivot2.plot(kind='bar', ax=ax2)
    ax2.set_title('Months to Profitability')
    ax2.set_ylabel('Months')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Total 3-Year Profit
    pivot3 = df_results.pivot(index='Growth Scenario', columns='Pricing Model', values='Total 3-Year Profit')
    pivot3.plot(kind='bar', ax=ax3)
    ax3.set_title('Total 3-Year Profit')
    ax3.set_ylabel('Total Profit ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Network Value vs Users
    for scenario in user_scenarios.keys():
        scenario_data = df_results[df_results['Growth Scenario'] == scenario]
        ax4.scatter(scenario_data['Final Users'], scenario_data['Final Network Value'], 
                   label=scenario, s=100, alpha=0.7)
    
    ax4.set_xlabel('Final Number of Users')
    ax4.set_ylabel('Final Network Value ($)')
    ax4.set_title('Network Value vs User Base')
    ax4.legend()
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return df_results

# Main execution
if __name__ == "__main__":
    print("🚀 Fresco AI Cost and Network Value Analysis")
    print("=" * 50)
    
    print("\n1. Cost Structure Analysis")
    analyze_cost_structure()
    
    print("\n2. Network Value Analysis")
    analyze_network_value()
    
    print("\n3. Pricing Model Comparison")
    compare_pricing_models()
    
    print("\n4. Break-even Analysis")
    break_even_analysis()
    
    print("\n5. Comprehensive Scenario Comparison")
    results_df = scenario_comparison()
    
    print("\n6. Interactive Parameter Analysis")
    print("Run the interactive_parameter_analysis() function to explore parameters dynamically")
    
    # Display summary table
    print("\n📊 Summary Results:")
    print(results_df.round(2).to_string(index=False))
    
    print("\n🎯 Key Insights:")
    print("- Network value grows as n*log(n), providing strong network effects")
    print("- Cost scaling is critical - infrastructure efficiency determines profitability")
    print("- Pricing model choice significantly impacts time to profitability")
    print("- User growth rate affects optimal pricing strategy")
    
    print("\n🔧 Interactive Features:")
    print("- Adjust cost per user, conversion rates, and pricing")
    print("- Test different network value multipliers")
    print("- Compare scenarios in real-time")
    
    # To run interactive analysis, uncomment the line below:
    # interactive_parameter_analysis()