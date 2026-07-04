#!/bin/bash

echo "🚀 Deploying Financial Intelligence System..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set up environment
echo "🔧 Setting up environment..."
cp .env.example .env 2>/dev/null || true

# Start main server
echo "🖥️ Starting main server..."
uvicorn main:app --host 0.0.0.0 --port 6900 &

# Start agent
echo "🤖 Starting SMT agent..."
uvicorn agent:agent_app --host 0.0.0.0 --port 6901 &

# Start scheduler
echo "⏰ Starting scheduler..."
python scheduler.py &

echo "✅ All services started!"
echo "   - Main API: http://localhost:6900"
echo "   - Agent API: http://localhost:6901"