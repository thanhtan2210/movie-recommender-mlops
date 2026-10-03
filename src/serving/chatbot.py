import os
import logging
import json
from typing import List, Dict, Any, Optional
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "chatbot.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_KEYS = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 6) if os.getenv(f"GROQ_API_KEY_{i}")]
if not API_KEYS and os.getenv("GROQ_API_KEY"):
    API_KEYS = [os.getenv("GROQ_API_KEY")]

current_key_idx = 0
client = Groq(api_key=API_KEYS[current_key_idx]) if API_KEYS else None

def rotate_key():
    global current_key_idx, client
    if len(API_KEYS) > 1:
        current_key_idx = (current_key_idx + 1) % len(API_KEYS)
        client = Groq(api_key=API_KEYS[current_key_idx])
        logger.info(f"Đã chuyển sang Groq API Key mới (index {current_key_idx})")
        return True
    return False

_SEARCH_ENGINE = None
def get_search_engine():
    global _SEARCH_ENGINE
    if _SEARCH_ENGINE is None:
        try:
            from src.serving.semantic_search import SemanticSearchEngine
            _SEARCH_ENGINE = SemanticSearchEngine()
            _SEARCH_ENGINE.load_table()
            logger.info("LanceDB Search Engine đã khởi tạo thành công.")
        except Exception as e:
            logger.error(f"Không thể khởi tạo LanceDB Search Engine: {e}. Sử dụng Mocking...")
            _SEARCH_ENGINE = "MOCK"
    return _SEARCH_ENGINE

# TOOLS DEFINITIONS (OPENAI COMPATIBLE)
def search_movies_by_description(query: str) -> str:
    engine = get_search_engine()
    if engine and engine != "MOCK":
        try:
            return json.dumps(engine.search_by_description(query, top_k=3), ensure_ascii=False)
        except Exception as e:
            logger.error(f"Lỗi search_by_description: {e}")
    return '{"results": [], "status": "mock"}'

def get_recommendations(title: str) -> str:
    engine = get_search_engine()
    if engine and engine != "MOCK":
        try:
            return json.dumps(engine.search_similar_movies_by_title(title, top_k=5), ensure_ascii=False)
        except Exception as e:
            logger.error(f"Lỗi get_recommendations: {e}")
    return '{"title": "' + title + '", "recommendations": []}'

def get_trending_by_rating(min_rating: float, min_votes: int) -> str:
    engine = get_search_engine()
    if engine and engine != "MOCK":
        try:
            return json.dumps(engine.get_trending_by_rating(min_rating, min_votes, top_k=5), ensure_ascii=False)
        except Exception as e:
            logger.error(f"Lỗi get_trending_by_rating: {e}")
    return '{"results": []}'

tools = [
    {"type": "function", "function": {"name": "search_movies_by_description", "description": "Tìm kiếm phim theo mô tả nội dung hoặc ngữ cảnh người dùng.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_recommendations", "description": "Lấy danh sách phim tương tự dựa trên tên bộ phim.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Tên bộ phim muốn lấy gợi ý tương tự"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "get_trending_by_rating", "description": "Lấy phim hay nhất theo rating tối thiểu và số vote.", "parameters": {"type": "object", "properties": {"min_rating": {"type": "number"}, "min_votes": {"type": "integer"}}, "required": ["min_rating", "min_votes"]}}}
]

# LEVEL 3: RAG Chatbot with Entity Memory
class MovieChatbot:
    def __init__(self):
        if not client:
            raise ValueError("Vui lòng cấu hình GROQ_API_KEY trong file .env")
        logger.info("Đang khởi tạo MovieChatbot (RAG + Memory) với Groq (Llama-3.3-70b-versatile)...")

    def chat(self, user_message: str, history: Optional[List[Dict[str, Any]]] = None, entity_memory: Optional[Dict[str, Any]] = None) -> str:
        engine = get_search_engine()
        context_str = ""
        
        # 1. RAG Retrieval Step
        if engine and engine != "MOCK":
            context_movies = engine.search_by_description(user_message, top_k=3)
            context_str = "THÔNG TIN TỪ DATABASE:\n"
            for m in context_movies:
                context_str += f"- Phim: {m['title']} | Thể loại: {m['genres']} | Điểm: {m['avg_rating']}⭐\n  Nội dung: {m['overview']}\n\n"

        # 2. Entity Memory Injection
        memory_str = ""
        if entity_memory:
            liked = list(entity_memory.get("liked_genres", []))
            if liked:
                memory_str = f"Sở thích của người dùng: {', '.join(liked)}\n"

        # 3. System Prompt Augmentation
        system_prompt = f"""Bạn là chuyên gia tư vấn phim AI. 
        TUYỆT ĐỐI KHÔNG tự bịa đặt thông tin. Chỉ sử dụng thông tin từ THÔNG TIN DATABASE cung cấp bên dưới. Nếu không có thông tin, hãy lịch sự trả lời: 'Tôi không có đủ dữ liệu về bộ phim này'. Trả lời ngắn gọn, thân thiện bằng tiếng Việt.
        {memory_str}
        {context_str}
        """

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history[-10:]:
                role = "assistant" if msg["role"] == "assistant" else "user"
                messages.append({"role": role, "content": msg["content"][:500]})
        messages.append({"role": "user", "content": user_message})

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=1000
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = globals()[func_name](**args)
                    messages.append(msg)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
                final_res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
                return final_res.choices[0].message.content
            return msg.content
            
        except RateLimitError:
            logger.warning("Quota exhausted, Rule-based Fallback Active...")
            if engine != "MOCK":
                res = engine.search_by_description(user_message, top_k=3)
                fallback_msg = "⚠️ Trợ lý AI đang nâng cấp hệ thống, nhưng dựa vào khóa tìm kiếm, tôi có vài gợi ý cho bạn:\n\n"
                for i, r in enumerate(res):
                    fallback_msg += f"{i+1}. **{r['title']}** ({r['genres']}) - ⭐ {r['avg_rating']}\n"
                return fallback_msg
            return "Hệ thống đang bảo trì, vui lòng thử lại sau."
        except Exception as e:
            logger.error(f"Lỗi xử lý Chatbot: {e}")
            return "Xin lỗi, kết nối API đang gặp sự cố nhỏ."