from __future__ import annotations

import os
import re
import datetime
import json
import urllib.request
import html
import smtplib
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Optional, Any, TypedDict, Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


# -------------------- Configuration --------------------

load_dotenv()

for _key in ("TAVILY_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
    if not os.getenv(_key):
        raise RuntimeError(f"Missing required environment variable: {_key}")

llm = init_chat_model(
    model="qwen/qwen3.8-27b",
    model_provider="groq",
    temperature=0.5,
    max_tokens=900,
)

IST = ZoneInfo("Asia/Kolkata")


def today_ist():
    """Return the current calendar date in Indian Standard Time."""
    return datetime.datetime.now(IST).date()


now = datetime.datetime.now(IST)
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H:%M:%S")
print(f"Date (IST): {current_date}")
print(f"Time (IST): {current_time}")

# GitHub runner IP geolocation is not the owner's location; use explicit settings.
city = os.getenv("CITY") or "Lucknow"
region = os.getenv("REGION") or "Uttar Pradesh"
country = os.getenv("COUNTRY") or "India"
lat = float(os.getenv("LAT") or "26.8467")
lon = float(os.getenv("LON") or "80.9462")
print(f"Configured Location: {city}, {region}, {country}")
print(f"Coordinates: Latitude {lat}, Longitude {lon}")

RECIPIENT = os.getenv("RECIPIENT_EMAIL") or "govind51n9h@gmail.com"
SENDER_NAME = os.getenv("SENDER_NAME") or "govinds-ai-agent"
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_RAG_ATTEMPTS = 4
INDEX_DIR = "faiss_index"
BANNER_URL: Optional[str] = None


# -------------------- Weather --------------------

@tool(
    "weather",
    description="Collects and returns info related to current weather of your location.",
)
def get_weather(
    lat: Optional[float],
    lon: Optional[float],
    headers: Optional[dict] = None,
) -> str:
    """Fetch weather data from Open-Meteo and return a formatted string."""
    if lat is None or lon is None:
        return "Could not retrieve latitude and longitude to fetch weather data."

    if headers is None:
        headers = {"User-Agent": "WeatherAgent/1.0"}

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&current_weather=true"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum"
        "&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
    except Exception as err:
        return f"Error fetching weather data: {err}"

    current = data.get("current_weather", {})
    daily = data.get("daily", {})
    today_date = daily.get("time", ["Today"])[0]
    temp_max = daily.get("temperature_2m_max", [None])[0]
    temp_min = daily.get("temperature_2m_min", [None])[0]
    precip = daily.get("precipitation_sum", [None])[0]

    lines = [
        f"--- Weather Forecast for Today ({today_date}) ---",
        f"Current Temperature: {current.get('temperature')}°C",
        f"Wind Speed: {current.get('windspeed')} km/h",
        f"Today's Max Temperature: {temp_max}°C",
        f"Today's Min Temperature: {temp_min}°C",
        f"Expected Precipitation: {precip} mm",
    ]
    return "\n".join(lines)


# -------------------- Tavily search --------------------

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    include_answer=True,
)


# -------------------- Indian history (updated filters) --------------------

def format_history_output(tavily_response: dict) -> str:
    """Extract specific date-linked events only when an Indian connection is present."""
    results = tavily_response.get("results", []) or []
    answer = str(tavily_response.get("answer", "") or "").strip()
    facts: list[dict[str, str]] = []
    seen: set[str] = set()

    generic_phrases = (
        "what happened today in history",
        "this day in history",
        "today in history",
        "historical events on this day",
        "history today",
        "events that happened today",
        "has seen remarkable historical events",
        "remarkable historical events",
        "read more",
        "click here",
        "latest news",
        "on this day",
    )

    event_signals = (
        "born", "birth", "died", "death", "anniversary", "founded",
        "established", "declared", "launched", "signed", "became",
        "independence", "battle", "revolution", "elected", "appointed",
        "opened", "inaugurated", "discovered", "passed away",
        "assassinated", "adopted", "formed", "created", "constitution",
        "movement", "war", "agreement", "treaty", "won", "started",
        "announced", "proclaimed", "government", "formation",
        "partition", "republic", "freedom", "conquered", "abdicated",
        "ratified", "recognized", "recognised",
    )

    # Heuristic signals for Indian relevance. These must appear in the
    # individual event candidate, not merely elsewhere on the page.
    indian_relevance_signals = (
        "india", "indian", "bharat", "bharatiya",
        "new delhi", "delhi", "mumbai", "bombay", "kolkata",
        "calcutta", "chennai", "madras", "lucknow", "kanpur",
        "agra", "patna", "jaipur", "hyderabad", "ahmedabad",
        "amritsar", "jallianwala", "plassey", "panipat",
        "kashmir", "ladakh", "punjab", "bengal", "gujarat",
        "maharashtra", "rajasthan", "uttar pradesh", "tamil nadu",
        "kerala", "karnataka", "assam", "odisha", "orissa",
        "goa", "sikkim", "manipur", "mughal india", "british india",
        "mahatma gandhi", "jawaharlal nehru", "sardar patel",
        "vallabhbhai patel", "subhas chandra bose", "bhagat singh",
        "chandra shekhar azad", "rani lakshmibai", "rani of jhansi",
        "ambedkar", "rabindranath tagore", "sarojini naidu",
        "bal gangadhar tilak", "dadabhai naoroji", "lala lajpat rai",
        "bipin chandra pal", "mangal pandey", "chhatrapati shivaji",
        "ashoka", "chandragupta maurya", "samudragupta", "akbar",
        "shah jahan", "tipu sultan", "bahadur shah zafar",
        "savitribai phule", "jyotirao phule", "swami vivekananda",
        "rajendra prasad", "indira gandhi", "atal bihari vajpayee",
        "abdul kalam", "indian national congress",
        "constituent assembly of india", "constitution of india",
        "supreme court of india", "reserve bank of india", "isro",
        "drdo", "indian parliament", "lok sabha", "rajya sabha",
        "indian national army", "azad hind", "quit india movement",
        "civil disobedience movement", "salt march",
        "non-cooperation movement", "swadeshi movement",
        "sepoy mutiny", "revolt of 1857", "partition of india",
        "indian independence", "republic of india",
        "indian space research organisation",
        "indian space research organization",
        "university of calcutta", "university of madras",
        "banaras hindu university", "aligarh muslim university",
    )

    def clean_candidate(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", str(value or ""))
        value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value)
        value = re.sub(r"^\s*[-*•]+\s*", "", value)
        return re.sub(r"\s+", " ", value).strip(" -|\t")

    def has_indian_relevance(value: str) -> bool:
        lower = clean_candidate(value).lower()
        return any(
            re.search(r"(?<![a-z])" + re.escape(signal) + r"(?![a-z])", lower)
            for signal in indian_relevance_signals
        )

    def is_concrete_event(value: str) -> bool:
        value = clean_candidate(value)
        lower = value.lower()

        if not 35 <= len(value) <= 600:
            return False
        if any(phrase in lower for phrase in generic_phrases):
            return False
        if lower.startswith(("http://", "https://", "source:", "read more")):
            return False

        has_year = bool(re.search(r"\b(?:1[5-9]\d{2}|20\d{2})\b", value))
        has_date = bool(re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b",
            lower,
        ))
        has_event_signal = any(signal in lower for signal in event_signals)

        return (
            (has_year or has_date)
            and has_event_signal
            and has_indian_relevance(value)
        )

    for item in results:
        title = clean_candidate(item.get("title", ""))
        content = str(item.get("content", "") or "").strip()
        candidates = [title]
        candidates.extend(re.split(r"(?<=[.!?])\s+|\n+", content))

        for candidate in candidates:
            candidate = clean_candidate(candidate)
            if not is_concrete_event(candidate):
                continue

            heading = (
                title
                if title
                and not any(p in title.lower() for p in generic_phrases)
                and has_indian_relevance(title)
                else candidate
            )

            key = re.sub(r"[^a-z0-9]", "", (heading + candidate).lower())
            if key in seen:
                continue
            seen.add(key)
            facts.append({"heading": heading[:180], "detail": candidate[:650]})
            break

        if len(facts) >= 5:
            break

    if not facts and answer:
        for candidate in re.split(r"(?<=[.!?])\s+|\n+", answer):
            candidate = clean_candidate(candidate)
            if is_concrete_event(candidate):
                facts.append({"heading": candidate[:180], "detail": candidate[:650]})
                break

    output_lines: list[str] = []
    for idx, fact in enumerate(facts, start=1):
        output_lines.extend([
            f"fact{idx}:",
            f"- heading: {fact['heading']}",
            f"- details: {fact['detail']}",
        ])

    output_lines.append("summary:")
    output_lines.append(
        "Specific Indian historical events extracted from search results."
        if facts
        else "No sufficiently specific Indian historical event was found in the search results."
    )
    return "\n".join(output_lines)


@tool(
    "search_india_history_on_date",
    description="Finds specific Indian historical events associated with a given date.",
)
def search_india_history_on_date(
    day: int,
    month_name: str,
    year: int = None,
    keywords: str = "major historic events facts",
):
    """Search Indian history and retry if no qualifying Indian event is found."""
    date_label = f"{day} {month_name}"

    if year:
        query = (
            f"Indian historical events on {date_label} {year}. Find specific named "
            "events in Indian history, Indian historical figures born or deceased "
            "on this date, Indian institutions, and anniversaries directly connected "
            "to India. Include the historical year and factual description. Exclude "
            f"unrelated international history and generic history pages. {keywords}"
        )
    else:
        query = (
            f"India historical events that occurred on {date_label}. Find specific "
            "named events in Indian history, Indian historical figures born or "
            "deceased on this date, and anniversaries directly connected to India. "
            "Include the historical year and factual description. Exclude unrelated "
            f"international history and generic history pages. {keywords}"
        )

    first_results = tavily_search_tool.invoke({"query": query})
    formatted = format_history_output(first_results)

    # A specific but unrelated international event does not count as success.
    if re.search(r"(?m)^fact\d+:$", formatted):
        print("[History] Found a qualifying Indian historical event in first search.")
        return formatted

    print("[History] No qualifying Indian event found; trying fallback query.")
    fallback_query = (
        f"Indian history timeline on {date_label}. Find named Indian historical "
        "events, notable Indian figures, Indian institutions, independence movement "
        "milestones, and events that occurred in India on this calendar date. "
        "Provide specific dates, historical years, and factual details. Exclude "
        "unrelated foreign events and generic history pages."
    )
    if year:
        fallback_query += f" Focus on {date_label} {year}."

    fallback_results = tavily_search_tool.invoke({"query": fallback_query})
    fallback_formatted = format_history_output(fallback_results)

    if not re.search(r"(?m)^fact\d+:$", fallback_formatted):
        print("[History] No qualifying Indian event found after fallback.")
    else:
        print("[History] Fallback found a qualifying Indian historical event.")

    return fallback_formatted


# -------------------- News --------------------

def format_news_output(tavily_response: dict) -> str:
    """Format Tavily results into news1..5 and a summary."""
    results = tavily_response.get("results", [])
    answer = tavily_response.get("answer", "")
    news_items = []

    for result in results:
        title = result.get("title", "").strip()
        content = result.get("content", "").strip()
        cleaned_title = re.sub(
            r"\s*[\\|-]\s*(The Hindu|The Indian Express|Times of India|Mathrubhumi).*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        sentences = [
            s.strip()
            for s in re.split(r"\n+|\.\s+", content)
            if len(s.strip()) > 30 and not s.startswith("#")
        ]
        if sentences:
            snippet = ". ".join(sentences[:2]) + "."
            news_items.append({"heading": cleaned_title, "content": snippet})
        if len(news_items) >= 5:
            break

    output_lines = []
    for idx, item in enumerate(news_items[:5], start=1):
        output_lines.extend([
            f"news{idx}:",
            f"heading: {item['heading']}",
            f"content: {item['content']}\n",
        ])

    output_lines.append("summary:")
    output_lines.append(
        answer or
        "Summary derived from top national, sports, political, and business news updates of the day."
    )
    return "\n".join(output_lines)


@tool(
    "fetch_previous_day_news",
    description="Collects and returns info related to previous day news wrt India.",
)
def fetch_previous_day_news(
    day: int,
    month: str | int,
    year: int,
    keywords: str = "India top news headlines politics business sports",
) -> str:
    """Fetch news for the day before the input date."""
    if isinstance(month, str):
        month_num = datetime.datetime.strptime(month[:3], "%b").month
    else:
        month_num = month

    target_date = datetime.date(year, month_num, day)
    previous_day_date = target_date - datetime.timedelta(days=1)
    formatted_prev_date = previous_day_date.strftime("%d %B %Y")
    print(
        f"Input Date: {target_date.strftime('%d %B %Y')} | "
        f"Fetching Previous Day News for: {formatted_prev_date}\n"
    )

    query = (
        f"What were the major news, headlines, and events in India on "
        f"{formatted_prev_date}? {keywords}"
    )
    results = tavily_search_tool.invoke({"query": query})
    return format_news_output(results)


# -------------------- Personal RAG --------------------

@tool(
    "rag",
    description="Collects and returns info related to owner personality or experience from vector db",
)
def search_docs(query: str) -> str:
    """Search the personal knowledge base for documented owner experiences."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# -------------------- HTML email sender --------------------

@tool(
    "send_email",
    description="Send an HTML-formatted morning email to the owner, optionally with a banner image.",
)
def send_email(
    receiver_email: str,
    sender_name: str,
    content: str,
    banner_url: Optional[str] = None,
) -> str:
    """Send HTML content as an email, with a plain-text fallback."""
    sender_email = os.getenv("gmail")
    sender_password = os.getenv("google_app_pass")

    if not sender_email or not sender_password:
        return "Error: Missing credentials. Set environment variables 'gmail' and 'google_app_pass'."

    message = MIMEMultipart("alternative")
    message["From"] = formataddr((sender_name, sender_email))
    message["To"] = receiver_email
    message["Subject"] = f"Good morning, Govind — {today_ist().strftime('%d %b %Y')}"

    text_content = html.unescape(re.sub(r"<[^>]+>", " ", content))
    text_content = re.sub(r"[ \t]+", " ", text_content)
    text_content = re.sub(r"\n\s*\n+", "\n\n", text_content).strip()
    message.attach(MIMEText(text_content, "plain", "utf-8"))

    banner_html = ""
    if banner_url:
        safe_url = html.escape(banner_url, quote=True)
        banner_html = (
            f'<img src="{safe_url}" alt="" '
            'style="display:block;max-width:100%;height:auto;border-radius:10px;margin:0 0 20px;">'
        )

    html_content = f"""<!doctype html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:24px;background:#f1f5f9;">
  <div style="max-width:680px;margin:0 auto;background:#ffffff;padding:28px;
              border-radius:14px;font-family:Arial,Helvetica,sans-serif;
              color:#172033;line-height:1.65;">
    {banner_html}
    {content}
    <hr style="border:0;border-top:1px solid #e2e8f0;margin:24px 0 12px;">
    <p style="color:#64748b;font-size:12px;margin:0;">
      Sent by {html.escape(sender_name)}
    </p>
  </div>
</body>
</html>"""
    message.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [receiver_email], message.as_string())
        return "Email sent successfully."
    except Exception as exc:
        return f"Email send failed: {type(exc).__name__}: {exc}"


# -------------------- Tool wrappers --------------------

@tool(
    "get_local_weather",
    description="Get current weather and today's forecast for the owner's configured location.",
)
def get_local_weather() -> str:
    return str(get_weather.invoke({"lat": lat, "lon": lon}))


@tool(
    "get_indian_history_for_today",
    description="Find notable Indian historical events on today's calendar date.",
)
def get_indian_history_for_today(
    keywords: str = "major historical events, notable births, anniversaries",
) -> str:
    today = today_ist()
    return str(search_india_history_on_date.invoke({
        "day": today.day,
        "month_name": today.strftime("%B"),
        "keywords": keywords,
    }))


@tool(
    "get_indian_news_from_yesterday",
    description="Find important Indian news from yesterday, not current breaking news.",
)
def get_indian_news_from_yesterday(
    keywords: str = "major India national, science, technology, business, sports and public-interest news",
) -> str:
    today = today_ist()
    return str(fetch_previous_day_news.invoke({
        "day": today.day,
        "month": today.month,
        "year": today.year,
        "keywords": keywords,
    }))


research_tools = [
    get_local_weather,
    get_indian_history_for_today,
    get_indian_news_from_yesterday,
]
research_llm = llm.bind_tools(research_tools)
research_tool_node = ToolNode(research_tools)

REQUIRED_TOOLS = {
    "get_local_weather",
    "get_indian_history_for_today",
    "get_indian_news_from_yesterday",
}


# -------------------- Graph state --------------------

class MorningState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    completed_tools: list[str]
    research: dict[str, str]
    theme: str
    rag_query: str
    rag_queries: list[str]
    rag_results: list[str]
    rag_attempts: int
    personal_evidence: str
    best_personal_evidence: str
    best_rag_relevance: float
    email_content: str
    result: str


# -------------------- Research phase --------------------

def research_node(state: MorningState) -> dict:
    """Fetch all required sources independently so one failure doesn't stop others."""
    research: dict[str, str] = {}
    research_jobs = [
        ("get_local_weather", get_local_weather, {}),
        ("get_indian_history_for_today", get_indian_history_for_today, {}),
        ("get_indian_news_from_yesterday", get_indian_news_from_yesterday, {}),
    ]

    for key, research_tool, args in research_jobs:
        try:
            result = research_tool.invoke(args)
            research[key] = str(result).strip() or f"{key} returned no data."
        except Exception as exc:
            research[key] = f"Research unavailable: {type(exc).__name__}: {exc}"
        print(f"[Research] {key}: {research[key][:500]}")

    return {
        "research": research,
        "completed_tools": sorted(REQUIRED_TOOLS),
    }


# -------------------- Theme and iterative RAG --------------------

def infer_theme_node(state: MorningState) -> dict:
    research = state.get("research", {})
    response = llm.invoke([
        SystemMessage(content="""You are an editorial planner. Identify a natural theme that may connect
the weather, one historical item, and yesterday's news. Do not force a connection.

Then create a specific query for the owner's personal RAG knowledge base. Search for any
documented experience, project, goal, challenge, achievement, habit, or lesson that could
connect even weakly to the theme. Personal information must come only from RAG.

Return valid JSON only: {"theme":"...", "rag_query":"...", "rationale":"..."}."""),
        HumanMessage(content=json.dumps(research, ensure_ascii=False)),
    ])

    raw = response.content if isinstance(response.content, str) else str(response.content)
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(match.group(0) if match else raw)
    except Exception:
        parsed = {
            "theme": "steady progress",
            "rag_query": "documented owner experiences projects goals challenges achievements habits lessons",
            "rationale": "Fallback because theme JSON could not be parsed.",
        }

    query = str(parsed.get("rag_query") or "documented owner experiences and lessons")
    return {
        "theme": str(parsed.get("theme") or "steady progress"),
        "rag_query": query,
        "rag_queries": [],
        "rag_attempts": 0,
        "rag_results": [],
        "best_personal_evidence": "",
        "best_rag_relevance": -1.0,
    }


def make_next_rag_query(state: MorningState) -> str:
    """Ask the LLM for a meaningfully different query each attempt."""
    prior_queries = state.get("rag_queries", [])
    prior_results = state.get("rag_results", [])
    if not prior_queries:
        return state.get("rag_query", "documented owner experiences and lessons")

    response = llm.invoke([
        SystemMessage(content="""You are improving retrieval from the owner's personal vector database.
Generate ONE concise search query for semantic retrieval. Search only for content that could
actually be in the owner's documents: personal experiences, projects, goals, applications,
learning, achievements, challenges, habits, setbacks, or lessons. The query can be weakly
related to the daily theme, but must be different in angle from prior queries. Do not answer
the query and do not invent personal facts. Return only the query, no quotes or explanation."""),
        HumanMessage(content=json.dumps({
            "theme": state.get("theme", ""),
            "weather_history_news": state.get("research", {}),
            "prior_queries": prior_queries,
            "latest_retrievals": prior_results[-2:],
        }, ensure_ascii=False)),
    ])

    query = response.content if isinstance(response.content, str) else str(response.content)
    query = query.strip().strip('"').strip()
    if not query or query in prior_queries:
        angles = [
            "owner's documented projects and what they learned",
            "owner's documented goals, obstacles, and persistence",
            "owner's documented habits, routines, and improvements",
            "owner's documented achievements, setbacks, and next steps",
        ]
        query = (
            f"{state.get('theme', 'today’s theme')}; "
            f"{angles[len(prior_queries) % len(angles)]}"
        )
    return query


def rag_retrieve_node(state: MorningState) -> dict:
    query = make_next_rag_query(state)
    try:
        result = str(search_docs.invoke({"query": query}))
    except Exception as exc:
        result = f"RAG retrieval error: {type(exc).__name__}: {exc}"

    prior_queries = list(state.get("rag_queries", []))
    prior_queries.append(query)
    results = list(state.get("rag_results", []))
    results.append(f"RAG QUERY:\n{query}\n\nRAG RETRIEVED DOCUMENTS:\n{result}")

    print(f"\n[RAG attempt {len(results)}/{MAX_RAG_ATTEMPTS}] Query: {query}")
    print(f"[RAG attempt {len(results)}] Retrieved output preview:\n{result[:1200]}\n")

    return {
        "rag_queries": prior_queries,
        "rag_query": query,
        "rag_results": results,
        "rag_attempts": len(results),
    }


def evaluate_rag_node(state: MorningState) -> dict:
    latest = state.get("rag_results", [""])[-1]
    response = llm.invoke([
        SystemMessage(content="""You select the best exact passage from retrieved personal RAG text.
The retrieval text is the ONLY permitted source for claims about the owner. Do not use
conversation memory, general knowledge, assumptions, or plausible inferences.

A very weak connection to the theme is acceptable. Pick the most useful owner-related passage
that actually occurs in the retrieval text. Return the exact substring verbatim, without
paraphrasing, repairing, or joining distant fragments. If the retrieval contains no personal
owner-related passage, return an empty quote. Score relevance from 0.0 to 1.0 only for ranking
quotes, not as a minimum threshold. Return valid JSON only:
{"quote":"exact substring or empty string","relevance":0.0,"reason":"brief"}.
Do not infer sensitive traits."""),
        HumanMessage(content=json.dumps({
            "theme": state.get("theme"),
            "query": state.get("rag_query"),
            "retrieval": latest,
        }, ensure_ascii=False)),
    ])

    raw = response.content if isinstance(response.content, str) else str(response.content)
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(match.group(0) if match else raw)
    except Exception:
        parsed = {"quote": "", "relevance": 0.0, "reason": "Invalid evaluator JSON"}

    quote = str(parsed.get("quote", "")).strip()
    if not quote or quote not in latest:
        quote = ""

    try:
        relevance = max(0.0, min(1.0, float(parsed.get("relevance", 0.0)))) if quote else -1.0
    except (TypeError, ValueError):
        relevance = 0.0 if quote else -1.0

    best_quote = state.get("best_personal_evidence", "")
    best_score = float(state.get("best_rag_relevance", -1.0))
    if quote and relevance > best_score:
        best_quote, best_score = quote, relevance

    print(f"[RAG evaluator] Found verified passage: {bool(quote)}; score={relevance}")
    if quote:
        print(f"[RAG evaluator] Exact selected passage: {quote[:600]}")

    return {
        "personal_evidence": best_quote,
        "best_personal_evidence": best_quote,
        "best_rag_relevance": best_score,
    }


def after_rag(state: MorningState) -> str:
    if state.get("rag_attempts", 0) < MAX_RAG_ATTEMPTS:
        return "rag_retrieve"
    return "write_email"


# -------------------- Deterministic HTML email writing --------------------

def _clean_text(value: Any, limit: int = 700) -> str:
    """Normalize source text for readable email output without inventing facts."""
    text_value = str(value or "").strip()
    text_value = re.sub(r"```(?:\w+)?|```", "", text_value)
    text_value = re.sub(r"\s+", " ", text_value)
    if len(text_value) > limit:
        text_value = text_value[:limit].rsplit(" ", 1)[0].rstrip() + "…"
    return text_value


def _first_history_item(raw: str) -> tuple[str, str]:
    """Extract an event and remove duplicated heading/detail text."""
    raw = str(raw or "")
    match = re.search(
        r"fact\d+:\s*\n\s*-\s*heading:\s*(.*?)\n"
        r"\s*-\s*details:\s*(.*?)(?=\n\s*fact\d+:|\n\s*summary:|$)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return (
            "No verified Indian event found",
            "Today's search did not return a sufficiently specific Indian historical event. Try again later.",
        )

    heading = _clean_text(match.group(1), 180)
    detail = _clean_text(match.group(2), 650)
    heading = re.sub(r"^#{1,6}\s*", "", heading).strip()
    detail = re.sub(r"^#{1,6}\s*", "", detail).strip()

    generic = (
        "what happened today in history",
        "this day in history",
        "has seen remarkable historical events",
        "no verified specific historical event",
        "no sufficiently specific indian historical event",
    )
    if (
        not heading
        or not detail
        or any(p in (heading + " " + detail).lower() for p in generic)
    ):
        return (
            "No verified Indian event found",
            "Today's search did not return a sufficiently specific Indian historical event. Try again later.",
        )

    if heading.casefold() == detail.casefold():
        detail = ""
    else:
        heading_normalized = re.sub(r"\s+", " ", heading).strip().rstrip(".!?")
        detail_normalized = re.sub(r"\s+", " ", detail).strip()
        if detail_normalized.casefold().startswith(heading_normalized.casefold()):
            remaining = detail_normalized[len(heading_normalized):]
            if not remaining or remaining[0] in ".!?: ":
                detail = remaining.lstrip(" .!?:").strip()

    if not detail:
        detail = "See the event heading above."

    return heading, detail


def _first_news_item(raw: str) -> tuple[str, str]:
    """Extract the first structured news item."""
    match = re.search(
        r"news\d+:\s*\n\s*heading:\s*(.*?)\n\s*content:\s*(.*?)(?=\n\s*news\d+:|\n\s*summary:|$)",
        raw or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        heading = _clean_text(match.group(1), 180)
        detail = _clean_text(match.group(2), 650)
        return heading or "Yesterday's India news", detail

    summary_match = re.search(r"summary:\s*(.*)", raw or "", flags=re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = _clean_text(summary_match.group(1), 650)
        if summary and "derived from top national" not in summary.lower():
            return "News summary", summary
    if raw and "unavailable" not in raw.lower() and "error" not in raw.lower():
        return "News update", _clean_text(raw, 650)
    return "News unavailable", "A reliable news item could not be retrieved today."


def _format_weather(raw: str) -> str:
    """Render weather tool output as safe readable lines."""
    if not raw or "error fetching" in raw.lower() or "could not retrieve" in raw.lower():
        return "Weather information is temporarily unavailable."
    lines = []
    for line in str(raw).splitlines():
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            lines.append(
                '<p style="margin:4px 0;"><strong>'
                + html.escape(label.strip())
                + ":</strong> "
                + html.escape(value.strip())
                + "</p>"
            )
        else:
            lines.append('<p style="margin:4px 0;">' + html.escape(line) + "</p>")
    return "".join(lines) or "Weather information is temporarily unavailable."


def write_email_node(state: MorningState) -> dict:
    """Build HTML deterministically so required sections cannot disappear."""
    research = state.get("research", {})
    theme = _clean_text(state.get("theme", "steady progress"), 160)
    evidence = str(state.get("best_personal_evidence", "") or "").strip()

    if evidence:
        personal_html = (
            '<section style="background:#f8fafc;border-left:4px solid #64748b;'
            'padding:14px 16px;margin:16px 0;border-radius:8px;">'
            '<h2 style="font-size:17px;margin:0 0 8px;color:#0f172a;">'
            'A note from your own journey</h2>'
            f'<blockquote style="margin:0;color:#334155;">{html.escape(evidence)}</blockquote>'
            '</section>'
        )
    else:
        personal_html = (
            '<section style="background:#f8fafc;border-left:4px solid #64748b;'
            'padding:14px 16px;margin:16px 0;border-radius:8px;">'
            '<h2 style="font-size:17px;margin:0 0 8px;color:#0f172a;">A thought for today</h2>'
            '<p style="margin:0;color:#334155;">Let today be about one useful step, not a perfect plan. '
            'Small, consistent actions can make progress visible over time.</p>'
            '</section>'
        )

    weather_html = _format_weather(
        research.get("get_local_weather", "Weather information was unavailable.")
    )
    history_heading, history_detail = _first_history_item(
        research.get("get_indian_history_for_today", "")
    )
    news_heading, news_detail = _first_news_item(
        research.get("get_indian_news_from_yesterday", "")
    )
    today_label = today_ist().strftime("%A, %d %B %Y")

    fragment = f"""
    <div style="font-family:Arial,Helvetica,sans-serif;color:#172033;line-height:1.6;">
      <header style="background:#0b2e6f;color:#ffffff;padding:22px 20px;border-radius:10px 10px 0 0;">
        <h1 style="font-size:24px;margin:0 0 6px;">Good morning, Govind ☀️</h1>
        <p style="margin:0;color:#e5edff;">{html.escape(theme)} — let's make today count.</p>
        <p style="font-size:12px;margin:8px 0 0;color:#dbeafe;">{html.escape(today_label)}</p>
      </header>

      {personal_html}

      <section style="border:1px solid #e2e8f0;padding:16px;margin:14px 0;border-radius:9px;">
        <h2 style="font-size:18px;color:#123b7a;margin:0 0 10px;">🌤️ Weather today · {html.escape(city)}</h2>
        {weather_html}
      </section>

      <section style="border:1px solid #e2e8f0;padding:16px;margin:14px 0;border-radius:9px;">
        <h2 style="font-size:18px;color:#123b7a;margin:0 0 8px;">📜 On this day in Indian history</h2>
        <p style="font-weight:bold;margin:0 0 5px;">{html.escape(history_heading)}</p>
        <p style="margin:0;">{html.escape(history_detail)}</p>
      </section>

      <section style="border:1px solid #e2e8f0;padding:16px;margin:14px 0;border-radius:9px;">
        <h2 style="font-size:18px;color:#123b7a;margin:0 0 8px;">📰 Yesterday's India news</h2>
        <p style="font-weight:bold;margin:0 0 5px;">{html.escape(news_heading)}</p>
        <p style="margin:0;">{html.escape(news_detail)}</p>
      </section>

      <p style="margin:20px 0 4px;">Wishing you a productive and resilient day ahead.</p>
    </div>
    """

    required_headings = (
        "On this day in Indian history",
        "Yesterday's India news",
    )
    missing = [heading for heading in required_headings if heading not in fragment]
    if missing:
        raise ValueError(f"Email is missing required sections: {missing}")

    print(
        "[Email validation] History section present; news section present; "
        f"HTML length={len(fragment)} characters."
    )
    return {"email_content": fragment}


# -------------------- Send / dry run --------------------

def send_email_node(state: MorningState) -> dict:
    content = state.get("email_content", "")
    if DRY_RUN:
        print("\n========== DRY RUN: EMAIL NOT SENT ==========\n")
        print(content)
        return {
            "result": "DRY RUN — no email sent. Review email_content, then set DRY_RUN=False.",
        }

    send_result = send_email.invoke({
        "receiver_email": RECIPIENT,
        "sender_name": SENDER_NAME,
        "content": content,
        "banner_url": BANNER_URL,
    })
    return {"result": str(send_result)}


# -------------------- Build graph --------------------

graph_builder = StateGraph(MorningState)
graph_builder.add_node("research", research_node)
graph_builder.add_node("infer_theme", infer_theme_node)
graph_builder.add_node("rag_retrieve", rag_retrieve_node)
graph_builder.add_node("evaluate_rag", evaluate_rag_node)
graph_builder.add_node("write_email", write_email_node)
graph_builder.add_node("send_email", send_email_node)

graph_builder.add_edge(START, "research")
graph_builder.add_edge("research", "infer_theme")
graph_builder.add_edge("infer_theme", "rag_retrieve")
graph_builder.add_edge("rag_retrieve", "evaluate_rag")
graph_builder.add_conditional_edges(
    "evaluate_rag",
    after_rag,
    {"rag_retrieve": "rag_retrieve", "write_email": "write_email"},
)
graph_builder.add_edge("write_email", "send_email")
graph_builder.add_edge("send_email", END)

morning_agent = graph_builder.compile()


# -------------------- Run --------------------

result = morning_agent.invoke({
    "messages": [
        HumanMessage(content=(
            "Start today's morning briefing. Collect current local weather, "
            "Indian historical information for today's calendar date, and Indian "
            "news from yesterday. Then search personal RAG and create the HTML email."
        ))
    ],
    "completed_tools": [],
    "research": {},
    "rag_results": [],
    "rag_queries": [],
})

print("\n--- RESULT ---")
print(result.get("result", "No result returned."))
print("\n--- VERIFIED PERSONAL EVIDENCE ---")
print(result.get("best_personal_evidence") or "[No personal passage was found in RAG results.]")
print("\n--- HTML EMAIL CONTENT ---")
print(result.get("email_content", ""))
