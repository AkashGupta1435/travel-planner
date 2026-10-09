"""
Travel Planner UI  -  same 4 agents and same flow as travel_planner.py,
shown in a web page instead of the terminal.

Run with:   streamlit run app.py
"""
import os
import re
from typing import Annotated

import autogen
import pyowm
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="centered")

AVATARS = {"Admin": "🧑‍💼", "Researcher": "🔎", "Planner": "🗺️", "Writer": "✍️"}
ROLES = {
    "Admin": "Starts the chat, runs the weather tool, stops at TERMINATE",
    "Researcher": "Asks for the weather of your city",
    "Planner": "Suggests packing and one activity",
    "Writer": "Formats the final itinerary",
}

# ------------------------------------------------------------------
# Sidebar: keys + how it works
# ------------------------------------------------------------------
# Keys come from the server (export on your Mac, or "Secrets" on Streamlit Cloud).
# They are never shown on the page, so visitors cannot see them.
SERVER_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
SERVER_OWM_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")  # optional

with st.sidebar:
    st.header("🔑 Keys")
    if SERVER_GROQ_KEY:
        st.success("Groq key is set on the server ✅")
        groq_key = SERVER_GROQ_KEY
    else:
        groq_key = st.text_input("Groq API key", type="password")

    if SERVER_OWM_KEY:
        st.success("Weather key is set on the server ✅")
        owm_key = SERVER_OWM_KEY
    else:
        owm_key = st.text_input("OpenWeatherMap key", type="password")

    if APP_PASSWORD:
        entered = st.text_input("Access code", type="password",
                                help="Ask the owner of this site for the code.")

    st.header("🤖 The team")
    for name, role in ROLES.items():
        st.markdown(f"{AVATARS[name]} **{name}** – {role}")

    st.caption("Flow: Admin → Researcher → Admin (runs tool) → Planner → Writer → Admin (stop)")


# ------------------------------------------------------------------
# Helper: show one chat message nicely
# ------------------------------------------------------------------
def show_message(msg):
    name = msg.get("name") or "Admin"
    with st.chat_message(name, avatar=AVATARS.get(name, "🤖")):
        st.markdown(f"**{name}**")

        # Researcher asked for a tool
        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                st.info(f"🔧 Asked to use **{fn.get('name')}** with `{fn.get('arguments')}`")

        # Admin ran the tool and returned the result
        if msg.get("tool_responses"):
            for resp in msg["tool_responses"]:
                st.success(f"🌦️ Tool result: {resp.get('content')}")
            return

        content = (msg.get("content") or "").strip()
        if content:
            st.markdown(content)


# ------------------------------------------------------------------
# The same lab logic, wrapped in a function so each click is a fresh run
# ------------------------------------------------------------------
def run_planner(city, groq_key, owm_key):
    # --- 1. CONFIG (same as the lab) ---
    config_list = [
        {
            "model": "openai/gpt-oss-120b",
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "cache_seed": None,
        }
    ]
    llm_config = {"config_list": config_list, "temperature": 0.2}

    # --- 2. TOOL (same as the lab) ---
    def get_weather(location: Annotated[str, "City, e.g. Chennai, IN"]) -> str:
        try:
            owm = pyowm.OWM(owm_key).weather_manager()
            w = owm.weather_at_place(location).weather
            return f"Weather in {location}: {w.temperature('celsius')['temp']}°C, {w.detailed_status}."
        except Exception as e:
            return f"Error: {str(e)}"

    # --- 3. AGENT SQUAD (same as the lab) ---
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="Admin. Execute tools. Reply TERMINATE when the Writer provides the plan.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or "").upper(),
    )
    researcher = autogen.AssistantAgent(
        name="Researcher",
        llm_config=llm_config,
        system_message="Researcher. Call get_weather for the requested city. Once data is received, stop.",
    )
    planner = autogen.AssistantAgent(
        name="Planner",
        llm_config=llm_config,
        system_message="""Travel Planner.
    1. Look at the weather data from the Researcher.
    2. Suggest specific packing (e.g., umbrella or light clothes).
    3. Suggest one indoor activity if it's raining, or one outdoor activity if it's clear.""",
    )
    writer = autogen.AssistantAgent(
        name="Writer",
        llm_config=llm_config,
        system_message="Writer. Format the final itinerary clearly and end with TERMINATE.",
    )
    autogen.agentchat.register_function(
        get_weather, caller=researcher, executor=user_proxy,
        name="get_weather", description="Get weather data",
    )

    # --- UI: show each new message as soon as it appears ---
    shown = {"count": 0}

    def show_new_messages(groupchat):
        for msg in groupchat.messages[shown["count"]:]:
            show_message(msg)
        shown["count"] = len(groupchat.messages)

    # --- 4. CUSTOM TRAVEL STATE MACHINE (same rules as the lab) ---
    def travel_logic(last_speaker, groupchat):
        show_new_messages(groupchat)  # UI addition: draw messages live
        messages = groupchat.messages
        if not messages:
            return researcher

        if last_speaker is user_proxy:
            last = messages[-1]
            # UI fix: in AutoGen 0.2 the tool result is marked by "tool_responses"
            if "tool_responses" in last or "Response from calling tool" in (last.get("content") or ""):
                st.caption("➡️ Data received. Moving to Planning...")
                return planner
            return researcher

        if last_speaker is researcher:
            return user_proxy if "tool_calls" in messages[-1] else planner

        if last_speaker is planner:
            st.caption("➡️ Plan generated. Moving to Writer...")
            return writer

        if last_speaker is writer:
            return user_proxy

        return "auto"

    # --- 5. RUN (same as the lab) ---
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, planner, writer],
        messages=[],
        max_round=12,
        speaker_selection_method=travel_logic,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_proxy.initiate_chat(
        manager,
        message=f"I am planning a trip to {city}. Get the weather and plan my trip.",
    )
    show_new_messages(groupchat)  # draw anything left after the chat ends
    return groupchat.messages


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
st.title("✈️ AI Travel Planner")
st.write("Type a city. Four AI agents will check the live weather and plan your trip.")

with st.form("trip"):
    city = st.text_input("Where are you going?", placeholder="e.g. Chennai, IN  or  London, GB")
    go = st.form_submit_button("Plan my trip", type="primary")

if go:
    if APP_PASSWORD and entered != APP_PASSWORD:
        st.error("Please enter the correct access code in the sidebar.")
        st.stop()
    if not groq_key or not owm_key:
        st.error("Please enter both keys in the sidebar.")
        st.stop()
    if not city.strip():
        st.warning("Please type a city first, like **Chennai, IN**.")
        st.stop()

    st.subheader("🗨️ Agent conversation")
    with st.status("Agents are working... (15–60 seconds)", expanded=True) as status:
        try:
            messages = run_planner(city.strip(), groq_key, owm_key)
            status.update(label="Done! The agents finished.", state="complete")
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

    # Weather card (from the tool result)
    weather_text = ""
    for m in messages:
        for r in m.get("tool_responses", []) or []:
            weather_text = r.get("content", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)°C,\s*(.+?)\.?$", weather_text)
    if match:
        c1, c2 = st.columns(2)
        c1.metric("🌡️ Temperature", f"{float(match.group(1)):.1f} °C")
        c2.metric("☁️ Sky", match.group(2).capitalize())
    elif weather_text.startswith("Error"):
        st.error(f"Weather tool failed: {weather_text}")

    # Final itinerary (last Writer message, without the word TERMINATE)
    writer_msgs = [m for m in messages if m.get("name") == "Writer" and m.get("content")]
    if writer_msgs:
        final = re.sub(r"\s*TERMINATE\s*", "", writer_msgs[-1]["content"]).strip()
        st.subheader("📋 Your itinerary")
        with st.container(border=True):
            st.markdown(final)
        st.download_button("⬇️ Download itinerary", final, file_name="itinerary.md")
    else:
        st.warning("The Writer did not produce a plan this time. Click **Plan my trip** again.")
