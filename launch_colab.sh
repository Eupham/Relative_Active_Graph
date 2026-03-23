#!/bin/bash
# Script to launch the Symbolic LLM Production UI from Google Colab with ngrok auth.

echo "⚡ Setting up Colab Production UI"
pip install -q streamlit pyngrok

# Provide Ngrok Token if not set
if [ -z "$NGROK_AUTH_TOKEN" ]; then
  read -p "Enter your Ngrok Auth Token (https://dashboard.ngrok.com/get-started/your-authtoken): " token
  export NGROK_AUTH_TOKEN=$token
fi

echo "🚀 Launching Streamlit Backend..."
streamlit run app.py &>/dev/null &

sleep 3
echo "✅ Done! Check the tunnel URL above to access the UI."
tail -f /dev/null
