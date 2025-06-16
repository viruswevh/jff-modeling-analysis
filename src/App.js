import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import './App.css';
import React, { useState, useEffect } from 'react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [analysisParams, setAnalysisParams] = useState({
    // Network Value Parameters
    network_value_multiplier: 100,
    n_users: 1000,
    
    // AI Cost Parameters
    base_cost_per_user: 0.1,
    infrastructure_base: 1000,
    infrastructure_scaling: 1.2,
    
    // Revenue Model Parameters
    pricing_model: 'freemium',
    subscription_price: 10,
    conversion_rate: 0.05,
    usage_price: 0.01,
    avg_usage_per_user: 100,
    revenue_per_user: 0,  // Calculated revenue per user
    total_revenue: 0,     // Total projected revenue
    
    // Market Analysis Parameters
    gdp: 1e12,
    population: 1e8,
    target_percentile: 50,
    price_to_income_ratio: 0.01,
    shape_parameter: 1.0,
    
    // Growth Scenario Parameters
    max_users: 100000,
    months: 36,
    
    // Profit Margin Parameters
    target_profit_margin: 25, // Target profit margin percentage
    operational_costs_ratio: 0.3 // Operational costs as ratio of revenue
  });

  const [selectedGraph, setSelectedGraph] = useState('cost_structure');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const chartRef = React.useRef(null);

  useEffect(() => {
    let chart = null;
    let frameId = null;

    const initChart = () => {
      if (!results || !chartRef.current) return;

      try {
        // Ensure any existing chart is destroyed
        if (chartRef.current.chart) {
          chartRef.current.chart.destroy();
        }

        // Create new chart instance
        chart = chartRef.current.chart;

        // Wait for the next frame to ensure the DOM is ready
        frameId = requestAnimationFrame(() => {
          try {
            if (chart && chart.canvas && chart.canvas.parentNode) {
              chart.update('none'); // Disable animation for the first update
            }
          } catch (error) {
            console.error('Error updating chart:', error);
            setError('Error displaying chart. Please try again.');
          }
        });
      } catch (error) {
        console.error('Error initializing chart:', error);
        setError('Error initializing chart. Please try again.');
      }
    };

    initChart();

    return () => {
      // Clean up
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
      try {
        if (chart && chart.destroy) {
          chart.destroy();
        }
      } catch (error) {
        console.error('Error destroying chart:', error);
      }
    };
  }, [results, selectedGraph]); // Also reinitialize when graph type changes

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, []);

  const formatNumber = (value, decimals = 2) => {
    if (typeof value !== 'number' || isNaN(value)) return '';
    return Number(value.toFixed(decimals));
  };

  const handleParamChange = (param, value) => {
    try {
      const newValue = typeof value === 'string' ? parseFloat(value) : value;
      if (!isNaN(newValue)) {
        setAnalysisParams(prev => ({
          ...prev,
          [param]: newValue
        }));
        setError(null); // Clear any previous errors
      }
    } catch (err) {
      setError(`Invalid value for ${param}: ${err.message}`);
    }
  };

  const validateParams = () => {
    if (selectedGraph === 'revenue_models') {
      const { pricing_model, subscription_price, conversion_rate, usage_price, avg_usage_per_user } = analysisParams;
      
      switch(pricing_model) {
        case 'subscription':
          if (subscription_price <= 0) {
            throw new Error('Subscription price must be greater than 0');
          }
          break;
        case 'usage_based':
          if (usage_price <= 0 || avg_usage_per_user <= 0) {
            throw new Error('Usage price and average usage must be greater than 0');
          }
          break;
        case 'hybrid':
          if (subscription_price <= 0 || usage_price <= 0 || avg_usage_per_user <= 0) {
            throw new Error('All pricing parameters must be greater than 0 for hybrid model');
          }
          break;
      }
      
      if (conversion_rate <= 0 || conversion_rate > 1) {
        throw new Error('Conversion rate must be between 0 and 1');
      }
    }
  };

  const validateParameters = () => {
    const { pricing_model, subscription_price, usage_price, avg_usage_per_user, conversion_rate } = analysisParams;
    
    if (pricing_model === 'subscription' && subscription_price <= 0) {
      throw new Error('Subscription price must be greater than 0 for subscription model');
    }
    
    if (pricing_model === 'usage_based' && usage_price <= 0) {
      throw new Error('Usage price must be greater than 0 for usage-based model');
    }
    
    if (pricing_model === 'hybrid') {
      if (subscription_price <= 0) {
        throw new Error('Subscription price must be greater than 0 for hybrid model');
      }
      if (usage_price <= 0) {
        throw new Error('Usage price must be greater than 0 for hybrid model');
      }
    }
    
    if (['usage_based', 'hybrid'].includes(pricing_model) && avg_usage_per_user <= 0) {
      throw new Error('Average usage per user must be greater than 0 for usage-based and hybrid models');
    }
    
    if (conversion_rate < 0 || conversion_rate > 1) {
      throw new Error('Conversion rate must be between 0 and 1');
    }
  };

  const runAnalysis = async () => {
    try {
      setError(null);
      
      // Validate parameters before making API call
      validateParameters();
      
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysisType: selectedGraph,
          params: {
            ...analysisParams,
            // Ensure required parameters are present for each model
            n_users: analysisParams.n_users || 1000,
            subscription_price: analysisParams.subscription_price || 0,
            conversion_rate: analysisParams.conversion_rate || 0,
            usage_price: analysisParams.usage_price || 0,
            avg_usage_per_user: analysisParams.avg_usage_per_user || 0
          }
        }),
      });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Server error: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        if (data.error) {
          throw new Error(data.error);
        }
        
        // Validate response data structure
        if (!data.graphData || !data.graphData.labels || !data.graphData.datasets) {
          throw new Error('Invalid data format received from server');
        }
        
        // Ensure datasets are not empty
        if (data.graphData.datasets.length === 0) {
          throw new Error('No data received for the selected parameters');
        }
        
        // Verify data points match labels
        const labelsLength = data.graphData.labels.length;
        const invalidDataset = data.graphData.datasets.find(ds => ds.data.length !== labelsLength);
        if (invalidDataset) {
          throw new Error(`Data mismatch in dataset: ${invalidDataset.label}`);
        }
        
        setResults(data);
    } catch (error) {
        console.error('Error in runAnalysis:', error);
        setError(error.message);
        setResults(null);
    }
};



  const renderInputs = () => {
    switch (selectedGraph) {
      case 'network_value':
        return (
          <div className="params-group">
            <h3>Network Value Parameters</h3>
            <div className="input-group">
              <label>Network Value Multiplier:</label>
              <input
                type="number"
                value={analysisParams.network_value_multiplier}
                onChange={(e) => handleParamChange('network_value_multiplier', parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Number of Users:</label>
              <input
                type="number"
                value={analysisParams.n_users}
                onChange={(e) => handleParamChange('n_users', parseInt(e.target.value))}
              />
            </div>
          </div>
        );

      case 'cost_structure':
        return (
          <div className="params-group">
            <h3>Cost Structure Parameters</h3>
            <div className="input-group">
              <label>Base Cost per User:</label>
              <input
                type="number"
                step="0.01"
                value={analysisParams.base_cost_per_user}
                onChange={(e) => handleParamChange('base_cost_per_user', parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Infrastructure Base Cost:</label>
              <input
                type="number"
                value={analysisParams.infrastructure_base}
                onChange={(e) => handleParamChange('infrastructure_base', parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Infrastructure Scaling:</label>
              <input
                type="number"
                step="0.1"
                value={analysisParams.infrastructure_scaling}
                onChange={(e) => handleParamChange('infrastructure_scaling', parseFloat(e.target.value))}
              />
            </div>
          </div>
        );

      case 'revenue_models':
        return (
          <div className="params-group">
            <h3>Revenue Model Parameters</h3>
            <div className="input-group">
              <label>Pricing Model:</label>
              <select
                value={analysisParams.pricing_model}
                onChange={(e) => {
                  const newModel = e.target.value;
                  const baseParams = {
                    pricing_model: newModel,
                    revenue_per_user: 0,
                    total_revenue: 0
                  };
                  
                  switch(newModel) {
                    case 'freemium':
                      Object.assign(baseParams, {
                        subscription_price: 10,
                        conversion_rate: 0.05,
                        usage_price: 0,
                        avg_usage_per_user: 0,
                        n_users: analysisParams.n_users || 1000 // Preserve or set default user count
                      });
                      break;
                    case 'subscription':
                      Object.assign(baseParams, {
                        subscription_price: 10,
                        conversion_rate: 1,
                        usage_price: 0,
                        avg_usage_per_user: 0,
                        n_users: analysisParams.n_users || 1000
                      });
                      break;
                    case 'usage_based':
                      Object.assign(baseParams, {
                        subscription_price: 0,
                        conversion_rate: 1,
                        usage_price: 0.01,
                        avg_usage_per_user: 100,
                        n_users: analysisParams.n_users || 1000
                      });
                      break;
                    case 'hybrid':
                      Object.assign(baseParams, {
                        subscription_price: 5,
                        conversion_rate: 0.1,
                        usage_price: 0.005,
                        avg_usage_per_user: 50,
                        n_users: analysisParams.n_users || 1000
                      });
                      break;
                    default:
                      throw new Error('Invalid pricing model selected');
                  }
                  
                  // Update all parameters at once to prevent intermediate invalid states
                  setAnalysisParams(prev => ({
                    ...prev,
                    ...baseParams
                  }));
                }}
              >
                <option value="freemium">Freemium</option>
                <option value="subscription">Subscription</option>
                <option value="usage_based">Usage Based</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </div>
            <div className="input-group">
              <label>Subscription Price:</label>
              <input
                type="number"
                min="0"
                step="0.01"
                disabled={analysisParams.pricing_model === 'usage_based'}
                value={analysisParams.subscription_price}
                onChange={(e) => handleParamChange('subscription_price', Math.max(0, parseFloat(e.target.value) || 0))}
              />
            </div>
            <div className="input-group">
              <label>Conversion Rate:</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={analysisParams.conversion_rate}
                onChange={(e) => {
                  const value = parseFloat(e.target.value);
                  if (value >= 0 && value <= 1) {
                    handleParamChange('conversion_rate', value);
                  }
                }}
              />
              <span style={{ marginLeft: '5px', fontSize: '0.8em', color: '#666' }}>
                (0-1)
              </span>
            </div>
            <div className="input-group">
              <label>Usage Price:</label>
              <input
                type="number"
                min="0"
                step="0.01"
                disabled={analysisParams.pricing_model === 'subscription' || analysisParams.pricing_model === 'freemium'}
                value={analysisParams.usage_price}
                onChange={(e) => handleParamChange('usage_price', Math.max(0, parseFloat(e.target.value) || 0))}
              />
            </div>
            <div className="input-group">
              <label>Average Usage per User:</label>
              <input
                type="number"
                min="0"
                step="1"
                disabled={analysisParams.pricing_model === 'subscription' || analysisParams.pricing_model === 'freemium'}
                value={analysisParams.avg_usage_per_user}
                onChange={(e) => handleParamChange('avg_usage_per_user', Math.max(0, parseInt(e.target.value) || 0))}
              />
            </div>
            <div className="input-group">
              <label>Target Profit Margin (%):</label>
              <input
                type="number"
                min="0"
                max="100"
                step="1"
                value={analysisParams.target_profit_margin}
                onChange={(e) => handleParamChange('target_profit_margin', Math.max(0, Math.min(100, parseFloat(e.target.value) || 0)))}
              />
            </div>
            <div className="input-group">
              <label>Operational Costs Ratio:</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={analysisParams.operational_costs_ratio}
                onChange={(e) => handleParamChange('operational_costs_ratio', Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
              />
              <span style={{ marginLeft: '5px', fontSize: '0.8em', color: '#666' }}>
                (0-1)
              </span>
            </div>
          </div>
        );

      case 'market_pricing':
        return (
          <div className="params-group">
            <h3>Market Analysis Parameters</h3>
            <div className="input-group">
              <label>GDP ($):</label>
              <input
                type="number"
                value={analysisParams.gdp}
                onChange={(e) => handleParamChange('gdp', parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Population:</label>
              <input
                type="number"
                value={analysisParams.population}
                onChange={(e) => handleParamChange('population', parseInt(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Target Percentile:</label>
              <input
                type="range"
                min="0"
                max="100"
                value={analysisParams.target_percentile}
                onChange={(e) => handleParamChange('target_percentile', parseFloat(e.target.value))}
              />
              <span>{analysisParams.target_percentile}%</span>
            </div>
            <div className="input-group">
              <label>Price to Income Ratio:</label>
              <input
                type="number"
                step="0.001"
                value={analysisParams.price_to_income_ratio}
                onChange={(e) => handleParamChange('price_to_income_ratio', parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Target Profit Margin (%):</label>
              <input
                type="number"
                min="0"
                max="100"
                step="1"
                value={analysisParams.target_profit_margin}
                onChange={(e) => handleParamChange('target_profit_margin', Math.max(0, Math.min(100, parseFloat(e.target.value) || 0)))}
              />
            </div>
          </div>
        );

      case 'user_growth':
        return (
          <div className="params-group">
            <h3>Growth Scenario Parameters</h3>
            <div className="input-group">
              <label>Maximum Users:</label>
              <input
                type="number"
                value={analysisParams.max_users}
                onChange={(e) => handleParamChange('max_users', parseInt(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Months:</label>
              <input
                type="number"
                value={analysisParams.months}
                onChange={(e) => handleParamChange('months', parseInt(e.target.value))}
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  // Initialize chart configuration
  const getChartConfig = (results) => ({
    data: {
      labels: results.graphData.labels,
      datasets: results.graphData.datasets.map(dataset => ({
        ...dataset,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 3,
        fill: dataset.fill !== undefined ? dataset.fill : false,
        backgroundColor: dataset.backgroundColor || 'rgba(52, 152, 219, 0.2)',
        borderColor: dataset.borderColor || '#3498db',
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1000,
        easing: 'easeOutQuart'
      },
      plugins: {
        title: {
          display: true,
          text: results.title,
          font: { 
            size: 18,
            family: "'Segoe UI', Roboto, sans-serif",
            weight: 'bold'
          },
          padding: { top: 10, bottom: 20 },
          color: '#2c3e50'
        },
        legend: {
          position: 'top',
          labels: { 
            usePointStyle: true,
            padding: 15,
            font: {
              family: "'Segoe UI', Roboto, sans-serif",
              size: 12
            },
            boxWidth: 8
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#333',
          bodyColor: '#666',
          borderColor: '#3498db',
          borderWidth: 1,
          padding: 12,
          cornerRadius: 6,
          titleFont: {
            family: "'Segoe UI', Roboto, sans-serif",
            size: 14,
            weight: 'bold'
          },
          bodyFont: {
            family: "'Segoe UI', Roboto, sans-serif",
            size: 13
          },
          displayColors: true,
          caretSize: 8,
          caretPadding: 8,
          boxShadow: '0 4px 10px rgba(0,0,0,0.1)'
        }
      },
      scales: getScaleConfig(selectedGraph),
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    }
  });

  const getScaleConfig = (graphType) => {
    const baseConfig = {
      x: {
        display: true,
        grid: { 
          display: true,
          color: 'rgba(0, 0, 0, 0.05)',
          drawBorder: false
        },
        ticks: {
          padding: 8,
          font: {
            family: "'Segoe UI', Roboto, sans-serif",
            size: 11
          },
          color: '#666'
        }
      },
      y: {
        display: true,
        position: 'left',
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
          drawBorder: false
        },
        ticks: {
          padding: 8,
          font: {
            family: "'Segoe UI', Roboto, sans-serif",
            size: 11
          },
          color: '#666'
        }
      }
    };

    switch(graphType) {
      case 'market_pricing':
        return {
          ...baseConfig,
          x: {
            ...baseConfig.x,
            title: { 
              display: true, 
              text: 'Target Percentile',
              font: {
                family: "'Segoe UI', Roboto, sans-serif",
                size: 13,
                weight: 'bold'
              },
              padding: { top: 10, bottom: 0 },
              color: '#2c3e50'
            }
          },
          y: {
            ...baseConfig.y,
            title: { 
              display: true, 
              text: 'Revenue ($)',
              font: {
                family: "'Segoe UI', Roboto, sans-serif",
                size: 13,
                weight: 'bold'
              },
              padding: { top: 0, bottom: 10 },
              color: '#2c3e50'
            },
            ticks: { 
              callback: value => '$' + value.toFixed(2),
              font: {
                family: "'Segoe UI', Roboto, sans-serif",
                size: 11
              },
              color: '#666'
            }
          }
        };
      case 'user_growth':
        return {
          ...baseConfig,
          x: {
            ...baseConfig.x,
            type: 'linear',
            title: { 
              display: true, 
              text: 'Months',
              font: {
                family: "'Segoe UI', Roboto, sans-serif",
                size: 13,
                weight: 'bold'
              },
              padding: { top: 10, bottom: 0 },
              color: '#2c3e50'
            }
          },
          y: {
            ...baseConfig.y,
            title: { 
              display: true, 
              text: 'Users',
              font: {
                family: "'Segoe UI', Roboto, sans-serif",
                size: 13,
                weight: 'bold'
              },
              padding: { top: 0, bottom: 10 },
              color: '#2c3e50'
            },
            ticks: { 
              callback: value => value.toLocaleString(undefined, {maximumFractionDigits: 2}),
              font: {
                family: "'Segoe UI', Roboto, sans-serif",
                size: 11
              },
              color: '#666'
            }
          }
        };
      default:
        return baseConfig;
    }
  };

  // Animation styles for staggered fade-in effect
  const fadeIn = (delay = 0) => ({
    animation: `fadeIn 0.5s ease-in-out ${delay}s both`
  });

  return (
    <div className="App">
      <header className="app-header">
        <img src="/logo.svg" alt="Jacques Fresco Foundation" className="app-logo" />
        <div>
          <h1 className="app-title">Fresco AI Analysis</h1>
          <p className="app-subtitle">Interactive modeling and analysis tools</p>
        </div>
      </header>
      
      <div className="main-container">
        <div className="controls-container" style={fadeIn(0.1)}>
          <div className="control-group" style={fadeIn(0.2)}>
            <h2>Graph Selection</h2>
            <div className="input-group">
              <label>Select Graph Type:</label>
              <select
                value={selectedGraph}
                onChange={(e) => setSelectedGraph(e.target.value)}
              >
                <option value="cost_structure">Cost Structure Analysis</option>
                <option value="network_value">Network Value Growth</option>
                <option value="revenue_models">Revenue Models Comparison</option>
                <option value="user_growth">User Growth Scenarios</option>
                <option value="market_pricing">Market Pricing Analysis</option>
              </select>
            </div>
          </div>

          {renderInputs()}

          <button className="run-button" onClick={runAnalysis} style={fadeIn(0.3)}>Run Analysis</button>
          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        <div className="graph-container" style={fadeIn(0.4)}>
          {results && (
            <div className="graph-section">
              <div className="graph" style={{ position: 'relative', minHeight: '400px' }}>
                {error ? (
                  <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center',
                    color: '#c62828'
                  }}>
                    <p>Error loading chart. Please try again.</p>
                  </div>
                ) : (
                  <Line
                    ref={chartRef}
                    {...getChartConfig(results)}
                    key={selectedGraph}
                  />
                )}
              </div>
              <div className="graph-description">
                {(() => {
                  switch(selectedGraph) {
                    case 'cost_structure':
                      return (
                        <>
                          <p>This model analyzes how costs scale with user growth, combining per-user costs and infrastructure overhead. The logarithmic scale shows the non-linear relationship between user base expansion and total costs.</p>
                          <div className="formula">
                            <strong>Formula:</strong><br />
                            Total Cost = (Base Cost × Number of Users) + (Infrastructure Base × Infrastructure Scaling<sup>log(Users)</sup>)<br />
                            Where:<br />
                            Base Cost = {analysisParams.base_cost_per_user.toFixed(2)} per user<br />
                            Infrastructure Base = ${analysisParams.infrastructure_base.toFixed(2)}<br />
                            Infrastructure Scaling = {analysisParams.infrastructure_scaling.toFixed(2)}
                          </div>
                        </>
                      );
                    case 'network_value':
                      return (
                        <>
                          <p>The network value model demonstrates how the platform's total value grows with its user base. It incorporates network effects where each additional user increases the value for all existing users.</p>
                          <div className="formula">
                            <strong>Formula:</strong><br />
                            Network Value = Network Multiplier × Users × log(Users)<br />
                            Where:<br />
                            Network Multiplier = ${analysisParams.network_value_multiplier.toFixed(2)}<br />
                            Users = {analysisParams.n_users}
                          </div>
                        </>
                      );
                    case 'revenue_models':
                      return (
                        <>
                          <p>Compare different revenue strategies: Freemium (free basic + premium features), Subscription (fixed recurring fee), Usage-based (pay-per-use), and Hybrid (combination). Each model's effectiveness varies with user behavior and market positioning.</p>
                          <div className="formula">
                            <strong>Formula:</strong><br />
                            {analysisParams.pricing_model === 'freemium' && (
                              <>Freemium Revenue = Users × Conversion Rate × Subscription Price<br />
                              Gross Profit = Revenue × (1 - Operational Costs Ratio)<br />
                              Required Revenue for Target Profit = Gross Profit ÷ (1 - Target Profit Margin ÷ 100)<br />
                              Where: Users = {analysisParams.n_users}, Conversion = {(analysisParams.conversion_rate * 100).toFixed(2)}%, Price = ${analysisParams.subscription_price.toFixed(2)}, Target Margin = {analysisParams.target_profit_margin}%, Op Costs = {(analysisParams.operational_costs_ratio * 100).toFixed(1)}%</>
                            )}
                            {analysisParams.pricing_model === 'subscription' && (
                              <>Subscription Revenue = Users × Subscription Price<br />
                              Gross Profit = Revenue × (1 - Operational Costs Ratio)<br />
                              Required Revenue for Target Profit = Gross Profit ÷ (1 - Target Profit Margin ÷ 100)<br />
                              Where: Users = {analysisParams.n_users}, Price = ${analysisParams.subscription_price.toFixed(2)}, Target Margin = {analysisParams.target_profit_margin}%, Op Costs = {(analysisParams.operational_costs_ratio * 100).toFixed(1)}%</>
                            )}
                            {analysisParams.pricing_model === 'usage_based' && (
                              <>Usage Revenue = Users × Average Usage × Usage Price<br />
                              Gross Profit = Revenue × (1 - Operational Costs Ratio)<br />
                              Required Revenue for Target Profit = Gross Profit ÷ (1 - Target Profit Margin ÷ 100)<br />
                              Where: Users = {analysisParams.n_users}, Avg Usage = {analysisParams.avg_usage_per_user}, Price = ${analysisParams.usage_price.toFixed(2)}, Target Margin = {analysisParams.target_profit_margin}%, Op Costs = {(analysisParams.operational_costs_ratio * 100).toFixed(1)}%</>
                            )}
                            {analysisParams.pricing_model === 'hybrid' && (
                              <>Hybrid Revenue = (Users × Subscription Price) + (Users × Average Usage × Usage Price)<br />
                              Gross Profit = Revenue × (1 - Operational Costs Ratio)<br />
                              Required Revenue for Target Profit = Gross Profit ÷ (1 - Target Profit Margin ÷ 100)<br />
                              Where: Users = {analysisParams.n_users}, Sub Price = ${analysisParams.subscription_price.toFixed(2)}, Usage = {analysisParams.avg_usage_per_user}, Usage Price = ${analysisParams.usage_price.toFixed(2)}, Target Margin = {analysisParams.target_profit_margin}%, Op Costs = {(analysisParams.operational_costs_ratio * 100).toFixed(1)}%</>
                            )}
                          </div>
                        </>
                      );
                    case 'user_growth':
                      return (
                        <>
                          <p>Visualize user acquisition scenarios over time, showing how different growth rates and market conditions affect the platform's user base expansion.</p>
                          <div className="formula">
                            <strong>Formula:</strong><br />
                            Users(t) = Max Users × (1 - e<sup>-rt</sup>)<br />
                            Where:<br />
                            Max Users = {analysisParams.max_users}<br />
                            Time Period = {analysisParams.months} months<br />
                            r = growth rate (calculated based on target timeline)
                          </div>
                        </>
                      );
                    case 'market_pricing':
                      return (
                        <>
                          <p>Analyze optimal pricing based on market demographics, showing revenue potential across different price points and target market segments.</p>
                          <div className="formula">
                            <strong>Formula:</strong><br />
                            Base Price = GDP per Capita × Price-to-Income Ratio × Income Distribution Factor<br />
                            Profit-Adjusted Price = Base Price ÷ (1 - Target Profit Margin ÷ 100)<br />
                            Where:<br />
                            GDP per Capita = ${(analysisParams.gdp / analysisParams.population).toFixed(2)}<br />
                            Price-to-Income Ratio = {analysisParams.price_to_income_ratio.toFixed(2)}<br />
                            Target Percentile = {analysisParams.target_percentile}%<br />
                            Target Profit Margin = {analysisParams.target_profit_margin}%
                          </div>
                        </>
                      );
                    default:
                      return '';
                  }
                })()}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;