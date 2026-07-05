# Financial Intelligence System

AI-powered financial analysis and market intelligence platform with SMT divergence detection.

## Features

- Real-time XAUUSD, XAGUSD, GBPUSD, EURUSD price monitoring
- SMT divergence detection across correlated assets
- Multi-timeframe analysis (30m, 1h, 4h, 1d, 1w)
- Email notifications for trading opportunities
- OpenBB Workspace integration with custom widgets

Screenshot include: 

<img width="1358" height="578" alt="image" src="https://github.com/user-attachments/assets/3c112206-6ae8-4748-82d3-dbe955ece14f" />


## Asset Coverage

| Group           | Base Asset | Correlated Assets      |
| :-------------- | :--------- | :--------------------- |
| Precious Metals | XAUUSD     | XAGUSD, XAUEUR, XAUGBP |
| Forex           | GBPUSD     | EURUSD                 |
| Futures         | Nasdaq     | ES, YM                 |

## Installation

```bash
git clone https://github.com/yourusername/financial-intelligence-system.git
cd financial-intelligence-system
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
uvicorn main:app --host 127.0.0.1 --port 6900 --reload
```
