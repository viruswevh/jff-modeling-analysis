# JFF Modeling and Analysis Tool

A comprehensive business modeling and analysis tool that integrates profit margin calculations with various business metrics. This tool helps analyze network effects, revenue models, market pricing, and cost structures while considering profit margins and operational costs.

## Features

- **Network Value Analysis**: Metcalfe's Law implementation with engagement metrics
- **Cost Structure Analysis**: AI infrastructure and variable cost modeling
- **Revenue Models Comparison**: Multiple revenue model analysis
- **Market Pricing Analysis**: GDP-based pricing with profit margin integration
- **Profit Margin Integration**: Target profit margin and operational cost calculations

## Mathematical Models & Formulas

### 1. Network Value Analysis
```
Network Value = N × M
Where:
N = Number of users
M = Value multiplier (default: $100)
```

### 2. AI Infrastructure Costs
```
Total Costs = VC + IC
Where:
VC (Variable Costs) = N × CPU
IC (Infrastructure Costs) = IB × N^(s-1)

N = Number of users
CPU = Base cost per user (default: $0.1)
IB = Infrastructure base cost (default: $1000)
s = Infrastructure scaling factor (default: 1.2)
```

### 3. Revenue Models

#### Freemium Model
```
Revenue = N × CR × SP
Where:
N = Number of users
CR = Conversion rate
SP = Subscription price
```

#### Subscription Model
```
Revenue = N × SP
Where:
N = Number of users
SP = Subscription price
```

#### Usage-Based Model
```
Revenue = N × UP × AU
Where:
N = Number of users
UP = Usage price
AU = Average usage per user
```

#### Hybrid Model
```
Revenue = (N × CR × SP) + (N × UP × AU)
Where:
N = Number of users
CR = Conversion rate
SP = Subscription price
UP = Usage price
AU = Average usage per user
```

#### Profit Calculations (All Models)
```
Gross Profit = Revenue × (1 - OCR)
Net Profit = Gross Profit × (TPM / 100)
Required Revenue = Gross Profit + Net Profit

Where:
OCR = Operational costs ratio
TPM = Target profit margin (%)
```

### 4. Market Pricing Analysis
```
Mean Income = GDP / Population
µ = ln(Mean Income) - (σ² / 2)
Target Income = lognorm.ppf(percentile, σ, exp(µ))
Base Price = (Target Income × 0.01) / 12
Profit-Adjusted Price = Base Price / (1 - TPM/100)
Market Size = Population × (1 - Target Percentile)

Where:
σ = Income distribution standard deviation (0.5)
TPM = Target profit margin (%)
```

## Dependencies

### Backend Dependencies
```bash
pip install flask==2.0.1 numpy==1.21.0 scipy==1.7.0 flask-cors==3.0.10
```

### Frontend Dependencies
```bash
npm install react@17.0.2 react-dom@17.0.2 chart.js@3.5.0 @material-ui/core@4.12.3
```

## Installation

### Backend Setup
1. Create a Python virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/MacOS
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Flask server:
```bash
python app.py
```

### Frontend Setup
1. Install Node.js dependencies:
```bash
npm install
```

2. Start the React development server:
```bash
npm start
```

## Usage

1. Access the application at `http://localhost:3000`
2. Select the analysis type from the dropdown menu
3. Input the required parameters:
   - For Network Value: Number of users
   - For Revenue Models: User count, pricing, conversion rates
   - For Market Pricing: GDP, population, target percentile
4. View the generated charts and analysis

## Development

### Project Structure
```
├── backend/
│   ├── app.py                 # Flask server
│   └── fresco_ai_model.py     # Core business logic
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   └── components/        # UI components
│   └── public/
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License