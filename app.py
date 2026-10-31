"""
Sensei4Stocks - Multi-Agent Stock Recommender
A professional Streamlit web application for AI-powered stock analysis.
"""

import os
import asyncio
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
from voice_generator import (
    generate_voice_output, 
    generate_individual_agent_audio,
    AGENT_VOICES,
    transcribe_audio
)
from streamlit_autorefresh import st_autorefresh
load_dotenv()

# Content length thresholds for filtering agent messages
# MIN_STREAMING_CONTENT_LENGTH: Minimum length to capture during streaming (filters out tool calls)
# Shorter threshold (50) used during streaming to capture intermediate analysis updates
MIN_STREAMING_CONTENT_LENGTH = 50
# MIN_FINAL_CONTENT_LENGTH: Minimum length for final message history (filters out handoffs)
# Longer threshold (100) used for final output to ensure only substantial analysis is shown
MIN_FINAL_CONTENT_LENGTH = 100

st.set_page_config(
    page_title="Sensei4Stocks | AI Stock Recommender",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* -------------------- GLOBAL -------------------- */
body, .stApp {
    background-color: rgba(10, 31, 68, 1);
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}

/* -------------------- LINKS -------------------- */
a, mark {
    color: #00e5e5;
}
a:hover {
    color: #00e5e5;
}

/* -------------------- MAIN CONTAINER -------------------- */
.main {
    padding: 1rem 2rem;
}

/* -------------------- HEADER -------------------- */
.header-container {
    background: linear-gradient(135deg, #0D9488 0%, #059669 50%, #10B981 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(13, 148, 136, 0.3);
}
.header-title {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}
.header-subtitle {
    color: #E0F2F1;
    font-size: 1.1rem;
    text-align: center;
    margin-top: 0.5rem;
}

/* -------------------- AGENT CARD -------------------- */
.agent-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    border-left: 4px solid #10B981;
}
.agent-name {
    font-weight: 600;
    color: #0D9488;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}
.agent-content {
    color: #333;
    line-height: 1.6;
}

/* -------------------- STATUS BADGES -------------------- */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
.status-running {
    background: #FEF3C7;
    color: #D97706;
}
.status-complete {
    background: #D1FAE5;
    color: #059669;
}

/* -------------------- SIDEBAR -------------------- */
.stSidebar, 
.stSidebar * {
    color: black !important;
}
.stSidebar .sidebar-content {
    background: #F0FDF4 !important;
}

/* -------------------- METRIC CARDS -------------------- */
.metric-card {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}

/* -------------------- BUTTONS -------------------- */
.stButton>button {
    background: linear-gradient(135deg, #0D9488 0%, #059669 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.4);
}

/* -------------------- PROGRESS -------------------- */
.progress-step {
    display: flex;
    align-items: center;
    padding: 0.5rem;
    margin: 0.25rem 0;
    color: black;
}
.progress-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 10px;
}
.dot-active { background: #10B981; }
.dot-pending { background: #E5E7EB; }

/* -------------------- INFO BOX -------------------- */
.info-box {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    color: black;
}

/* -------------------- FOOTER -------------------- */
.footer {
    text-align: center;
    color: #6B7280;
    padding: 2rem;
    margin-top: 2rem;
    border-top: 1px solid #E5E7EB;
}

/* -------------------- VOICE INPUT SECTION -------------------- */
.voice-section {
    background-color: rgba(10, 31, 68, 1) !important; /* dark blue */
    padding: 1rem;
    border-radius: 12px;
    margin: 1rem 0;
}


/* -------------------- AGENT CIRCLES -------------------- */
.agent-circles-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    padding: 2rem;
    margin: 1.5rem 0;
    background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    border-radius: 15px;
    border: 1px solid #A7F3D0;
    flex-wrap: wrap;
}
.agent-circle {
    display: flex;
    flex-direction: column;
    align-items: center;
    transition: all 0.3s ease;
}
.agent-circle-icon {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    background: white;
    border: 3px solid #E5E7EB;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.agent-circle-label {
    margin-top: 0.75rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #6B7280;
    text-align: center;
    max-width: 100px;
}
.agent-circle.active .agent-circle-icon {
    border-color: #10B981;
    box-shadow: 0 0 0 4px rgba(16,185,129,0.3),0 8px 25px rgba(16,185,129,0.4);
    transform: scale(1.1);
    animation: pulse 1.5s ease-in-out infinite;
}
.agent-circle.active .agent-circle-label {
    color: #059669;
    font-weight: 600;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 4px rgba(16,185,129,0.3),0 8px 25px rgba(16,185,129,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(16,185,129,0.2),0 8px 30px rgba(16,185,129,0.5); }
}

/* -------------------- VOICE OUTPUT -------------------- */
.voice-output-section {
    background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #A7F3D0;
}
.voice-output-title {
    text-align: center;
    color: #0D9488;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
.voice-play-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #0D9488 0%, #059669 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0 auto;
    box-shadow: 0 4px 15px rgba(13,148,136,0.3);
}
.voice-play-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(13,148,136,0.4);
}
.speaking-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: #059669;
    font-weight: 500;
    margin-top: 1rem;
    font-size: 0.95rem;
}
.speaking-indicator .sound-wave {
    display: inline-flex;
    align-items: center;
    gap: 2px;
}
.speaking-indicator .sound-wave span {
    display: inline-block;
    width: 3px;
    background: #10B981;
    animation: soundWave 0.5s ease-in-out infinite;
}
.speaking-indicator .sound-wave span:nth-child(1) { height: 8px; animation-delay: 0s; }
.speaking-indicator .sound-wave span:nth-child(2) { height: 12px; animation-delay: 0.1s; }
.speaking-indicator .sound-wave span:nth-child(3) { height: 16px; animation-delay: 0.2s; }
.speaking-indicator .sound-wave span:nth-child(4) { height: 12px; animation-delay: 0.3s; }
.speaking-indicator .sound-wave span:nth-child(5) { height: 8px; animation-delay: 0.4s; }
@keyframes soundWave {
    0%,100% { transform: scaleY(1); }
    50% { transform: scaleY(1.5); }
}

/* -------------------- MOBILE RESPONSIVE -------------------- */
@media (max-width: 768px) {
    .agent-circles-container {
        flex-direction: row;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem;
        padding: 1rem;
    }
    .agent-circle {
        margin-bottom: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


def reset_voice_state():
    """Reset all voice playback-related session state variables."""
    st.session_state.analysis_complete = False
    st.session_state.voice_playing = False
    st.session_state.voice_playback_complete = False
    st.session_state.agent_audio_files = {}
    st.session_state.current_playing_agent = None
    st.session_state.playback_index = 0
    st.session_state.agents_to_play = []


def format_agent_name(name):
    """Format agent name for display."""
    return name.replace('_', ' ').title()


def get_agent_display_info():
    """Get display information for all agents including supervisor."""
    return {
        "supervisor": {"icon": "👨‍💼", "label": "Supervisor"},
        "stock_finder_agent": {"icon": "🔍", "label": "Stock Finder"},
        "market_data_agent": {"icon": "📊", "label": "Market Data"},
        "news_analyst_agent": {"icon": "📰", "label": "News Analyst"},
        "price_recommender_agent": {"icon": "💰", "label": "Recommender"},
    }





def render_agent_circles(active_agent=None):
    """Render agent circles with optional highlighting for active agent."""
    agents = get_agent_display_info()
    circles_html = '<div class="agent-circles-container">'
    
    for agent_key, agent_info in agents.items():
        active_class = "active" if agent_key == active_agent else ""
        circles_html += f'<div class="agent-circle {active_class}" id="circle-{agent_key}">'
        circles_html += f'<div class="agent-circle-icon">{agent_info["icon"]}</div>'
        circles_html += f'<div class="agent-circle-label">{agent_info["label"]}</div>'
        circles_html += '</div>'
    
    circles_html += '</div>'
    return circles_html


def create_agent_card(agent_name, content, icon="🤖"):
    """Create a styled agent card."""
    formatted_name = format_agent_name(agent_name)
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-name">{icon} {formatted_name}</div>
        <div class="agent-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def parse_message_content(msg):
    """Extract content from message object."""
    if hasattr(msg, 'content'):
        return msg.content
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content']
    return str(msg)


def get_agent_name(msg):
    """Extract agent name from message object."""
    if hasattr(msg, 'name'):
        return msg.name
    elif isinstance(msg, dict) and 'name' in msg:
        return msg['name']
    return None


async def run_stock_analysis(query, progress_container, status_text, result_container):
    """Run the multi-agent stock analysis."""
    
    agent_icons = {
        "stock_finder_agent": "🔍",
        "market_data_agent": "📊",
        "news_analyst_agent": "📰",
        "price_recommender_agent": "💰",
        "supervisor": "👨‍💼"
    }
    
    try:
        status_text.text("🔄 Initializing AI agents...")
        
        client = MultiServerMCPClient(
            {
                "bright_data": {
                    "command": "npx",
                    "args": ["@brightdata/mcp"],
                    "env": {
                        "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                        "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                    },
                    "transport": "stdio",
                },
            }
        )
        tools = await client.get_tools()
        model = init_chat_model(
            model="meta/llama-3.3-70b-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            model_provider="openai",
            base_url="https://integrate.api.nvidia.com/v1",
            max_tokens=1000
        )
        
        status_text.text("🤖 Creating specialized agents...")
        
        stock_finder_agent = create_react_agent(
            model, tools,
            prompt="""You are a stock research analyst specializing in the Indian Stock Market (NSE). 
            Your task is to select 2 promising, actively traded NSE-listed stocks for short term trading 
            (buy/sell) based on recent performance, news buzz, volume or technical strength.
            Avoid penny stocks and illiquid companies.
            
            MANDATORY TOOL USAGE:
            - You MUST use the web scraping tools to fetch real data from financial websites like NSE India, 
              MoneyControl, Economic Times, or TradingView.
            - NEVER generate fictional stock names, prices, or metrics. Every piece of data must come from tool results.
            - If a tool call fails, retry with different parameters or a different URL. Do not make up data.
            - If you cannot retrieve real data after multiple attempts, explicitly state "DATA UNAVAILABLE" 
              and explain the issue - do NOT provide hypothetical alternatives.
            
            DATA VERIFICATION:
            - Cite the exact source URL for each piece of information you provide.
            - Only report data that appears in tool responses - do not extrapolate or invent.
            
            FORMAT YOUR RESPONSE:
            - Start by explaining what you searched for and how (e.g., "I searched for top gaining NSE stocks...")
            - Present each stock with clear reasoning:
              Stock 1: [Name] ([Ticker])
              - Source: [URL where you found this data]
              - Selection criteria: [why this stock stood out]
              - Key metrics: [volume, recent performance, etc.]
              
              Stock 2: [Name] ([Ticker])
              - Source: [URL where you found this data]  
              - Selection criteria: [why this stock stood out]
              - Key metrics: [volume, recent performance, etc.]
            
            Do NOT introduce yourself. Focus on your findings and methodology.""",
            name="stock_finder_agent"
        )
        
        market_data_agent = create_react_agent(
            model, tools,
            prompt="""You are a market data analyst for Indian stocks listed on NSE. Given a list of 
            stock tickers, your task is to gather recent market information for each stock.
            
            MANDATORY TOOL USAGE:
            - You MUST use web scraping tools to fetch real market data from financial websites like 
              NSE India, MoneyControl, Yahoo Finance India, or TradingView.
            - NEVER invent or estimate prices, volumes, or technical indicators. All data must come from tool results.
            - If a tool call fails, retry with different URLs (e.g., try MoneyControl if NSE fails).
            - If you cannot retrieve real data after multiple attempts, explicitly state "DATA UNAVAILABLE FOR [METRIC]"
              rather than providing made-up numbers.
            
            DATA VERIFICATION:
            - Cite the exact source URL for each data point you report.
            - Only report metrics that appear verbatim in tool responses.
            - Do not calculate or derive metrics unless explicitly shown in source data.
            
            FORMAT YOUR RESPONSE:
            - Start by explaining how you gathered the data (e.g., "I pulled data from NSE/MoneyControl...")
            - Present for EACH stock:
              
              [Stock Name] ([Ticker]):
              - Data Source: [URL]
              - Current Price: ₹XXX | Previous Close: ₹XXX | Change: X%
              - Volume: XXX (compared to average: higher/lower)
              - 7-day trend: [up/down X%]
              - Technical signals: RSI at XX, trading [above/below] 50-day MA (if available)
              - Assessment: [bullish/bearish/neutral] because [reason based on fetched data]
            
            Do NOT introduce yourself. Focus on the data and what it indicates.""",
            name="market_data_agent"
        )
        
        news_analyst_agent = create_react_agent(
            model, tools,
            prompt="""You are a financial news analyst. Given the names or the tickers of Indian NSE 
            listed stocks, your job is to find and analyze recent news.
            
            MANDATORY TOOL USAGE:
            - You MUST use web scraping tools to search and retrieve real news from financial news sites like 
              Economic Times, MoneyControl, Business Standard, LiveMint, or Reuters India.
            - NEVER fabricate news headlines, dates, or sources. Every headline must come from tool results.
            - If a search returns no results, try different search terms or news sources.
            - If you cannot find real news after multiple attempts, explicitly state "NO RECENT NEWS FOUND"
              rather than inventing headlines.
            
            DATA VERIFICATION:
            - Cite the exact source URL for each news headline you report.
            - Only report headlines that appear verbatim in tool responses.
            - Include the actual publication date from the source, not estimated dates.
            
            FORMAT YOUR RESPONSE:
            - Start by explaining your search approach (e.g., "I searched financial news sites for...")
            - Present for EACH stock:
              
              [Stock Name] ([Ticker]):
              - Key headlines found:
                • [Headline 1] - [Date] - [Source URL] - [Positive/Negative/Neutral]
                • [Headline 2] - [Date] - [Source URL] - [Positive/Negative/Neutral]
              - Overall sentiment: [Positive/Negative/Neutral]
              - Impact on stock: [How this news could affect price]
            
            Do NOT introduce yourself. Focus on news findings and sentiment analysis.""",
            name="news_analyst_agent"
        )
        
        price_recommender_agent = create_react_agent(
            model, tools,
            prompt="""You are a trading strategy advisor for the Indian Stock Market. You are given 
            market data and news analysis from other agents.
            
            MANDATORY DATA VERIFICATION:
            - Base your recommendations ONLY on the actual data provided by other agents.
            - NEVER invent prices, targets, or stop-loss levels. Use only prices mentioned in the provided data.
            - If the current price is ₹100, your target and stop-loss must be calculated from that real price.
            - If critical data (like current price) is missing or marked as "DATA UNAVAILABLE", 
              explicitly state "CANNOT PROVIDE RECOMMENDATION - INSUFFICIENT DATA" for that stock.
            
            RECOMMENDATION RULES:
            - Entry Price: Must match the current market price from the Market Data Agent's report.
            - Target Price: Calculate based on technical levels mentioned in the data, or use a reasonable 
              percentage (5-10%) above entry if no technical targets are available.
            - Stop Loss: Calculate based on support levels mentioned, or use a reasonable percentage 
              (3-5%) below entry if no support levels are available.
            - Clearly state if any calculation is an estimate due to limited data.
            
            FORMAT YOUR RESPONSE:
            - Start by summarizing what data you analyzed (e.g., "Based on the market data showing X and news indicating Y...")
            - Present for EACH stock:
              
              [Stock Name] ([Ticker]):
              - Data Source Summary: [Briefly cite what data you're using from other agents]
              - Recommendation: BUY / SELL / HOLD
              - Entry Price: ₹XXX (current market price from data)
              - Target Price: ₹XXX (X% upside potential) - [basis for target]
              - Stop Loss: ₹XXX (X% downside risk) - [basis for stop loss]
              - Reasoning: [Combine technical signals + news sentiment + risk assessment]
            
            Do NOT introduce yourself. Focus on actionable recommendations with clear reasoning.""",
            name="price_recommender_agent"
        )
        
        def trim_messages(messages, keep_last=3):
            return messages[-keep_last:]
        
        supervisor = create_supervisor(
            model=init_chat_model(
                model="meta/llama-3.3-70b-instruct",
                api_key=os.getenv("NVIDIA_API_KEY"),
                model_provider="openai",
                base_url="https://integrate.api.nvidia.com/v1",
                max_tokens=1000
            ),
            agents=[stock_finder_agent, market_data_agent, news_analyst_agent, price_recommender_agent],
            prompt=(
                "You are the Supervisor coordinating a stock analysis workflow with four expert agents.\n\n"
                "WORKFLOW - Call ALL agents in sequence:\n"
                "1. stock_finder_agent - identifies 2 promising NSE stocks\n"
                "2. market_data_agent - gathers market data and technical indicators\n"
                "3. news_analyst_agent - analyzes recent news and sentiment\n"
                "4. price_recommender_agent - gives buy/sell recommendations\n\n"
                "CRITICAL DATA INTEGRITY RULES:\n"
                "- Call one agent at a time, wait for their full response before calling the next\n"
                "- ALL four agents MUST provide their analysis - do not skip any\n"
                "- VERIFY that each agent's response includes source URLs or citations for their data\n"
                "- If an agent's response contains phrases like 'DATA UNAVAILABLE' or 'NO RESULTS FOUND',\n"
                "  acknowledge this in your final verdict rather than filling in with made-up data\n"
                "- REJECT any response that appears to contain hypothetical or example data without sources\n"
                "- If data quality is poor, note this limitation in the final verdict\n\n"
                "YOUR FINAL VERDICT (after all agents have completed their analysis):\n"
                "Do NOT introduce yourself. Provide a consolidated summary:\n\n"
                "📊 FINAL VERDICT\n"
                "═══════════════════════════\n\n"
                "**[Stock 1 Name] ([TICKER])**\n"
                "• Why Selected: [key reason from stock finder]\n"
                "• Price Data: ₹XXX (X% change), Volume [high/normal/low] - Source: [URL]\n"
                "• News Impact: [sentiment and key news] - Source: [URL]\n"
                "• Recommendation: [BUY/SELL/HOLD] | Entry: ₹XXX | Target: ₹XXX | Stop Loss: ₹XXX\n\n"
                "**[Stock 2 Name] ([TICKER])**\n"
                "• Why Selected: [key reason from stock finder]\n"
                "• Price Data: ₹XXX (X% change), Volume [high/normal/low] - Source: [URL]\n"
                "• News Impact: [sentiment and key news] - Source: [URL]\n"
                "• Recommendation: [BUY/SELL/HOLD] | Entry: ₹XXX | Target: ₹XXX | Stop Loss: ₹XXX\n\n"
                "📋 Data Quality Note: [Note any data limitations or unavailable information]\n\n"
                "⚠️ Disclaimer: This analysis is for educational purposes. Always do your own research."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()
        
        status_text.text("🚀 Running analysis...")
        
        agent_outputs = {}
        agent_all_content = {} 
        current_chunk = None
        
        for chunk in supervisor.stream(
            {
                "messages": trim_messages([
                    {
                        "role": "user",
                        "content": query,
                    }
                ])
            },
        ):
            current_chunk = chunk
            
            if isinstance(chunk, tuple):
                ns, chunk_data = chunk
                if len(ns) == 0:
                    continue
                graph_id = ns[-1].split(":")[0]
            else:
                chunk_data = chunk
            
            for node_name, node_update in chunk_data.items():
                if node_name in agent_icons:
                    icon = agent_icons.get(node_name, "🤖")
                    status_text.text(f"{icon} {format_agent_name(node_name)} is working...")
                    
                    messages = convert_to_messages(node_update.get("messages", []))
                    if messages:
                        for msg in messages:
                            content = parse_message_content(msg)
                            if content and isinstance(content, str) and content.strip():
                                if len(content.strip()) > MIN_STREAMING_CONTENT_LENGTH:
                                    if node_name not in agent_all_content:
                                        agent_all_content[node_name] = []
                                    agent_all_content[node_name].append(content)
 
                        if node_name in agent_all_content and agent_all_content[node_name]:
                            agent_outputs[node_name] = agent_all_content[node_name][-1]
        
        status_text.text("✅ Analysis complete!")
        
        final_message_history = []
        if current_chunk and "supervisor" in current_chunk:
            final_message_history = current_chunk["supervisor"].get("messages", [])

        agent_messages = []
        supervisor_final_message = None
        for msg in final_message_history:
            agent_name = get_agent_name(msg)
            content = parse_message_content(msg)

            if agent_name and content and isinstance(content, str) and content.strip():
                if len(content.strip()) > MIN_FINAL_CONTENT_LENGTH and agent_name in AGENT_VOICES:
                    agent_messages.append((agent_name, content))
                    agent_outputs[agent_name] = content
                    if agent_name == "supervisor":
                        supervisor_final_message = content

        st.session_state.agent_outputs = agent_outputs
        st.session_state.agent_messages = agent_messages
        st.session_state.analysis_complete = True
        
        with result_container:
            st.markdown("### 📋 Analysis Results")
            
            if agent_outputs:
                tabs = st.tabs([f"{agent_icons.get(name, '🤖')} {format_agent_name(name)}" 
                               for name in agent_outputs.keys()])
                
                for tab, (agent_name, content) in zip(tabs, agent_outputs.items()):
                    with tab:
                        st.markdown(content)
        
        return agent_outputs
        
    except Exception as e:
        status_text.text("❌ Error occurred")
        st.error(f"An error occurred: {str(e)}")
        return None


def main():
    import time
    
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📈 Sensei4Stocks</h1>
        <p class="header-subtitle">AI-Powered Multi-Agent Stock Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        
        st.markdown("### 📖 About")
        st.markdown("""
        **Sensei4Stocks** uses a team of AI agents to analyze Indian NSE stocks:
        
        🔍 **Stock Finder** - Identifies promising stocks
        
        📊 **Market Data Analyst** - Gathers technical data
        
        📰 **News Analyst** - Analyzes recent news
        
        💰 **Price Recommender** - Provides trading advice
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        <small>This tool is for educational purposes only. 
        Always do your own research before making investment decisions. 
        Past performance does not guarantee future results.</small>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Start Your Analysis")

        if 'query_text' not in st.session_state:
            st.session_state.query_text = "Find 2 promising NSE stocks and provide detailed analysis with buy/sell recommendations"
       
        if 'last_audio_processed' not in st.session_state:
            st.session_state.last_audio_processed = None
        if 'transcription_in_progress' not in st.session_state:
            st.session_state.transcription_in_progress = False

        if 'auto_run_analysis' not in st.session_state:
            st.session_state.auto_run_analysis = False
        
        st.markdown("#### 🎤 Voice Input")
        
        st.markdown("""
        <div class="info-box">
            <strong>🎙️ Record your query:</strong> Click the microphone icon to start/stop recording your stock analysis request.
        </div>
        """, unsafe_allow_html=True)
        
        audio_bytes = None
        recorder_available = False
        
        try:
            from audio_recorder_streamlit import audio_recorder
            st.markdown('<div class="voice-section style="background-color:#0A1F44; padding:1rem; border-radius:12px;">', unsafe_allow_html=True)
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e53935",
                neutral_color="#10B981",
                icon_size="2x",
                pause_threshold=2.0,  
            )
            recorder_available = True
            st.markdown('</div>', unsafe_allow_html=True)
        except ImportError:
            try:
                from st_audiorec import st_audiorec
                audio_bytes = st_audiorec()
                recorder_available = True
                
            except ImportError:
                pass
        
        if not recorder_available:
            st.warning("⚠️ Audio recording not available. Please type your query below.")
        else:
            st.info("💡 **Note:** Microphone access is required. Please allow permissions when prompted by your browser.")

        if audio_bytes is not None and len(audio_bytes) > 0:
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            
            if audio_hash != st.session_state.last_audio_processed:
                st.audio(audio_bytes, format="audio/wav")
                
                with st.spinner("🎙️ Transcribing your voice input..."):
                    st.session_state.transcription_in_progress = True
                    transcribed_text, error = transcribe_audio(audio_bytes)
                    st.session_state.transcription_in_progress = False
                    
                    st.session_state.last_audio_processed = audio_hash
                    
                    if transcribed_text:
                        st.success(f"✅ Transcribed: \"{transcribed_text}\"")
                        st.session_state.query_text = transcribed_text
                        st.session_state.auto_run_analysis = True
                        with st.spinner("🚀 Starting automatic stock analysis..."):
                            import time
                            time.sleep(1) 
                        st.rerun()
                    elif error:
                        st.warning(f"⚠️ Could not transcribe audio: {error}")
            else:
                st.audio(audio_bytes, format="audio/wav")
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.info("ℹ️ Audio transcribed. Record new audio or click 'Reset' to re-transcribe.")
                with col_btn:
                    if st.button("🔄 Reset", key="reset_audio"):
                        st.session_state.last_audio_processed = None
                        st.rerun()
            
        with st.expander("🔧 Troubleshooting: Voice input not working?"):
            st.markdown("""
            **If recording doesn't work:**
            
            1. **Check browser permissions:** Allow microphone access for this site
            2. **Use HTTPS:** Voice recording requires a secure connection (HTTPS) or localhost
            3. **Check microphone:** Ensure your microphone is connected and working
            4. **Try a different browser:** Chrome and Firefox work best
            5. **Refresh the page:** Sometimes a page refresh helps reset the audio context
            
            **Alternative:** You can always type your query in the text input below.
            """)
        
        st.markdown("---")
        
        st.markdown("#### ⌨️ Text Input")
        
        query = st.text_area(
            "Enter your analysis request:",
            value=st.session_state.query_text,
            height=100,
            help="Describe what kind of stock analysis you want the AI agents to perform",
            key="query_input"
        )
        
        if query != st.session_state.query_text:
            st.session_state.query_text = query
    
    with col2:
        st.markdown("### 🤖 Agent Pipeline")
        
        st.markdown("""
        <div style="background: #F0FDF4; padding: 1rem; border-radius: 10px; border: 1px solid #A7F3D0;">
            <div class="progress-step">
                <span style="font-size: 1.2rem;">1️⃣ </span>   Stock Discovery
            </div>
            <div class="progress-step">
                <span style="font-size: 1.2rem;">2️⃣ </span>   Market Data
            </div>
            <div class="progress-step">
                <span style="font-size: 1.2rem;">3️⃣ </span>   News Analysis
            </div>
            <div class="progress-step">
                <span style="font-size: 1.2rem;">4️⃣ </span>   Recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    run_analysis = st.button("🔍 Run Stock Analysis", type="primary", use_container_width=True)
    
    if st.session_state.get('auto_run_analysis', False):
        run_analysis = True
        st.session_state.auto_run_analysis = False  
    
    if run_analysis:
        
        reset_voice_state()
        analysis_query = st.session_state.query_text
        
        if not os.getenv("NVIDIA_API_KEY"):
            st.error("❌ NVIDIA_API_KEY is required. Please set it in your environment variables.")
            return
        
        if not os.getenv("BRIGHT_DATA_API_TOKEN"):
            st.error("❌ BRIGHT_DATA_API_TOKEN is required. Please set it in your environment variables.")
            return
        
        progress_container = st.container()
        status_text = st.empty()
        result_container = st.container()
        
        with progress_container:
            with st.spinner(""):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        run_stock_analysis(analysis_query, progress_container, status_text, result_container)
                    )
                finally:
                    loop.close()

    st.markdown("---")
    st.markdown("### 🔊 Voice Output & Agent Visualization")
    
    if 'voice_playing' not in st.session_state:
        st.session_state.voice_playing = False
    if 'current_playing_agent' not in st.session_state:
        st.session_state.current_playing_agent = None
    if 'agent_audio_files' not in st.session_state:
        st.session_state.agent_audio_files = {}
    if 'playback_index' not in st.session_state:
        st.session_state.playback_index = 0
    if 'agents_to_play' not in st.session_state:
        st.session_state.agents_to_play = []
    
    active_agent = st.session_state.current_playing_agent
    st.markdown(render_agent_circles(active_agent), unsafe_allow_html=True)
    
    analysis_complete = st.session_state.get('analysis_complete', False)
    agent_messages = st.session_state.get('agent_messages', [])
    
    if not analysis_complete:
        st.info("🎯 Run a stock analysis to enable voice output. The agent circles above will highlight as each agent speaks.")
    else:
    
        if not st.session_state.voice_playing and not st.session_state.get('voice_playback_complete', False):
            st.session_state.voice_playing = True
            st.session_state.playback_index = 0
    
            if agent_messages:
                with st.spinner("Generating voice for each agent..."):
                    audio_files, error = generate_individual_agent_audio(agent_messages, ".")
                    if audio_files:
                        st.session_state.agent_audio_files = audio_files
                        st.session_state.agents_to_play = list(audio_files.keys())
                        st.session_state.audio_durations = {}
                        for agent_name, file_path in audio_files.items():
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path)
                                estimated_duration = max(5, file_size / 48000)  
                                st.session_state.audio_durations[agent_name] = estimated_duration
                    elif error:
                        st.error(f"❌ {error}")
                        st.session_state.voice_playing = False
            else:
                st.warning("No agent messages available for voice output.")
                st.session_state.voice_playing = False

    if st.session_state.voice_playing and st.session_state.agent_audio_files:
        agents_to_play = st.session_state.agents_to_play
        playback_index = st.session_state.playback_index
        
        if playback_index < len(agents_to_play):
            current_agent = agents_to_play[playback_index]
            audio_file = st.session_state.agent_audio_files.get(current_agent)
            
            st.session_state.current_playing_agent = current_agent

            agent_info = get_agent_display_info().get(current_agent, {"icon": "🤖", "label": format_agent_name(current_agent)})
            st.markdown(f"""
            <div class="speaking-indicator">
                <div class="sound-wave">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
                <span>{agent_info["icon"]} {agent_info["label"]} is presenting their analysis...</span>
            </div>
            """, unsafe_allow_html=True)
            
 
            if audio_file and os.path.exists(audio_file):
                try:
                    with open(audio_file, "rb") as f:
                        audio_data = f.read()
                    
   
                    st.audio(audio_data, format="audio/wav", autoplay=True)
                    
                    audio_start_key = f"audio_start_{current_agent}"
                    if audio_start_key not in st.session_state:
                        st.session_state[audio_start_key] = time.time()
                    
                    duration = st.session_state.audio_durations.get(current_agent, 10)
                    elapsed = time.time() - st.session_state[audio_start_key]
                    
                    if elapsed >= duration:
                        del st.session_state[audio_start_key]
                        st.session_state.playback_index += 1
                        st.rerun()
                    else:
                        remaining = int(duration - elapsed)
                        st.caption(f"⏱️ Next agent in ~{remaining} seconds...")
                        refresh_interval = min(3000, max(1000, remaining * 500))
                        st_autorefresh(interval=refresh_interval, limit=None, key=f"audio_refresh_{playback_index}")
                    
                except IOError as e:
                    st.error(f"Error reading audio file: {e}")
        else:
            st.success("✅ Voice output complete!")
            st.session_state.voice_playing = False
            st.session_state.current_playing_agent = None
            st.session_state.playback_index = 0
            st.session_state.voice_playback_complete = True
    elif st.session_state.get('voice_playback_complete', False):
        st.success("✅ Voice output complete!")
    
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ by Shreya Kashikar</p>
        <p><small>© 2025 Sensei4Stocks | All Rights Reserved</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()