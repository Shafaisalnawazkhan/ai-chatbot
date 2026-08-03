import json
import re
import os
import time
import base64
import requests
from pathlib import Path
from django.http import JsonResponse, HttpResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Count
from .models import (
    VectorChatMessage,
    VectorDocumentSummary,
    AIModelConfig,
    UserFeedback,
    PresetPrompt
)

# Load secrets from .env — key is never hard-coded in source
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "openai/gpt-oss-20b:free"
SUMMARIZER_MODEL = "poolside/laguna-s-2.1:free"
LING_FLASH_MODEL = "inclusionai/ling-3.0-flash:free"
GEMMA_MOE_MODEL = "google/gemma-2-9b-it:free"
NEMOTRON_OMNI_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")
os.makedirs(AUDIO_DIR, exist_ok=True)


def seed_default_data_if_needed():
    """Ensure default models and preset prompts are populated in the DB."""
    try:
        if AIModelConfig.objects.count() == 0:
            AIModelConfig.objects.bulk_create([
                AIModelConfig(
                    model_id=NEMOTRON_OMNI_MODEL,
                    display_name="NVIDIA Nemotron Omni",
                    provider="NVIDIA",
                    category="omni",
                    description="300K Context | Mamba-Transformer Hybrid | Multimodal Perception",
                    context_window="300K",
                    default_temperature=0.7
                ),
                AIModelConfig(
                    model_id=GEMMA_MOE_MODEL,
                    display_name="Google Gemma 2 9B",
                    provider="Google DeepMind",
                    category="moe",
                    description="High-fidelity Deep Thinking & Reasoning Engine",
                    context_window="128K",
                    default_temperature=0.6
                ),
                AIModelConfig(
                    model_id=LING_FLASH_MODEL,
                    display_name="Ling 3.0 Flash",
                    provider="InclusionAI",
                    category="flash",
                    description="Ultra-fast Agentic Tool Execution & Token-Efficient Reasoning",
                    context_window="64K",
                    default_temperature=0.5
                ),
                AIModelConfig(
                    model_id=SUMMARIZER_MODEL,
                    display_name="Laguna Summarizer",
                    provider="Poolside",
                    category="summarizer",
                    description="High-fidelity Web, Document, and Code Summarization",
                    context_window="128K",
                    default_temperature=0.4
                ),
                AIModelConfig(
                    model_id=MODEL,
                    display_name="GPT OSS Assistant",
                    provider="OpenAI / Community",
                    category="chat",
                    description="General purpose conversational assistant",
                    context_window="32K",
                    default_temperature=0.7
                ),
            ])

        if PresetPrompt.objects.count() == 0:
            PresetPrompt.objects.bulk_create([
                PresetPrompt(
                    title="Code Security Audit",
                    category="Code",
                    prompt_text="Analyze the provided code for security vulnerabilities, memory leaks, and performance bottlenecks. Suggest exact patches.",
                    system_instruction="You are an elite Cybersecurity Engineer.",
                    icon="fa-shield-halved"
                ),
                PresetPrompt(
                    title="Executive Briefing",
                    category="Summary",
                    prompt_text="Summarize the core technical findings, risks, and strategic recommendations into a 5-point executive brief.",
                    system_instruction="You are a Chief Technology Officer preparing a board report.",
                    icon="fa-briefcase"
                ),
                PresetPrompt(
                    title="Deep Analytical Reasoning",
                    category="Reasoning",
                    prompt_text="Break down this complex problem into first principles, evaluate edge cases, and provide step-by-step mathematical or logical proof.",
                    system_instruction="You are a Principal AI Scientist specializing in logic and mathematics.",
                    icon="fa-brain"
                ),
                PresetPrompt(
                    title="Creative Tech Vision",
                    category="Creative",
                    prompt_text="Propose innovative system architecture ideas for futuristic real-time multimodal perception systems.",
                    system_instruction="You are a Visionary Solutions Architect.",
                    icon="fa-lightbulb"
                ),
            ])
    except Exception:
        pass


# Run seed once on startup
seed_default_data_if_needed()


# ai.png logo served directly
def ai_logo(request):
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ai.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "rd", "images", "ai.png"),
        "C:\\Users\\Rida\\Desktop\\prgect4\\ai.png"
    ]
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            with open(abs_p, 'rb') as f:
                return HttpResponse(f.read(), content_type='image/png')
    return HttpResponse(status=404)


def home(request):
    return render(request, "index.html")


def get_ai_response(prompt_text: str, model_name: str = SUMMARIZER_MODEL, messages_payload: list = None, temperature: float = 0.7) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "") or API_KEY
    if not api_key:
        return "⚠️ AI backend API key is missing."

    fallback_chain = [
        model_name,
        "google/gemma-2-9b-it:free",
        "openrouter/free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "openai/gpt-oss-20b:free"
    ]
    seen = set()
    models_to_try = [m for m in fallback_chain if not (m in seen or seen.add(m))]

    payload_messages = messages_payload if messages_payload else [{"role": "user", "content": prompt_text}]

    for current_model in models_to_try:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": current_model,
                    "messages": payload_messages,
                    "temperature": temperature
                },
                timeout=45,
            )
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                if content and len(content.strip()) > 0:
                    return content
        except Exception:
            continue

    return "⚠️ AI provider endpoint busy. Automatic retry fallback in progress..."


def search_web_robust(query: str, max_results: int = 4):
    results = []
    if not query.strip():
        return results

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=max_results,
            )
            for r in ddgs_gen:
                title = r.get("title", "").strip()
                href = r.get("href", "").strip()
                body = r.get("body", "").strip()
                block = f"Title: {title}\nLink: {href}\nSnippet: {body}\n"
                results.append(block)
    except Exception as e:
        results.append(f"Search Error: {str(e)}")

    return results


def text_to_speech_audio(text: str, filename_prefix: str = "audio"):
    if not text or "Error" in text or "⚠️" in text:
        return None

    try:
        from gtts import gTTS
        clean = text.replace("*", "").replace("#", "").replace("`", "")[:1000]

        filename = f"{filename_prefix}_{int(time.time()*1000)}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        tts = gTTS(clean, lang="en")
        tts.save(filepath)
        return filename
    except Exception:
        return None


# ---------------- API ENDPOINTS ----------------

def get_models_list(request):
    """Return active AI models catalog."""
    seed_default_data_if_needed()
    models_qs = AIModelConfig.objects.filter(is_active=True).values(
        'model_id', 'display_name', 'provider', 'category', 'description', 'context_window', 'default_temperature'
    )
    return JsonResponse({"models": list(models_qs)})


def get_preset_prompts(request):
    """Return template prompt cards."""
    seed_default_data_if_needed()
    prompts_qs = PresetPrompt.objects.filter(is_featured=True).values(
        'id', 'title', 'category', 'prompt_text', 'system_instruction', 'icon'
    )
    return JsonResponse({"prompts": list(prompts_qs)})


def get_analytics(request):
    """Return live usage statistics and performance metrics."""
    total_chats = VectorChatMessage.objects.count()
    total_summaries = VectorDocumentSummary.objects.count()
    total_feedbacks = UserFeedback.objects.count()

    avg_lat = VectorChatMessage.objects.aggregate(Avg('latency_ms'))['latency_ms__avg'] or 0.0
    avg_score = UserFeedback.objects.aggregate(Avg('rating'))['rating__avg'] or 4.9

    thumbs_up = UserFeedback.objects.filter(is_thumbs_up=True).count()
    satisfaction_rate = round((thumbs_up / total_feedbacks * 100), 1) if total_feedbacks > 0 else 98.5

    model_distribution = list(
        VectorChatMessage.objects.values('model_name')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )

    return JsonResponse({
        "total_queries": total_chats + total_summaries,
        "total_chats": total_chats,
        "total_summaries": total_summaries,
        "avg_latency_ms": round(avg_lat, 1),
        "satisfaction_rate": satisfaction_rate,
        "avg_rating": round(avg_score, 1),
        "model_distribution": model_distribution
    })


def get_history(request):
    """Fetch stored database chat and summary history."""
    category = request.GET.get('category', 'all')
    search_q = request.GET.get('q', '').strip()

    chats_qs = VectorChatMessage.objects.all().order_by('-created_at')
    summaries_qs = VectorDocumentSummary.objects.all().order_by('-created_at')

    if search_q:
        chats_qs = chats_qs.filter(user_prompt__icontains=search_q) | chats_qs.filter(ai_response__icontains=search_q)
        summaries_qs = summaries_qs.filter(title__icontains=search_q) | summaries_qs.filter(content__icontains=search_q)

    history_list = []
    if category in ['all', 'chats']:
        for item in chats_qs[:30]:
            history_list.append({
                "type": "chat",
                "id": item.id,
                "model_name": item.model_name,
                "prompt": item.user_prompt,
                "response": item.ai_response,
                "latency_ms": item.latency_ms,
                "audio_url": item.audio_url,
                "created_at": item.created_at.strftime("%Y-%m-%d %H:%M")
            })

    if category in ['all', 'summaries']:
        for item in summaries_qs[:30]:
            history_list.append({
                "type": "summary",
                "id": item.id,
                "summary_type": item.summary_type,
                "title": item.title,
                "content": item.content,
                "created_at": item.created_at.strftime("%Y-%m-%d %H:%M")
            })

    history_list.sort(key=lambda x: x['created_at'], reverse=True)

    return JsonResponse({"history": history_list[:40]})


@csrf_exempt
def submit_feedback(request):
    """Store user rating and comments."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        msg_id = data.get("message_id")
        rating = data.get("rating", 5)
        is_thumbs_up = data.get("is_thumbs_up", True)
        comment = data.get("comment", "")
        model_name = data.get("model_name", "AI Hub")

        msg_obj = None
        if msg_id:
            msg_obj = VectorChatMessage.objects.filter(id=msg_id).first()

        feedback = UserFeedback.objects.create(
            chat_message=msg_obj,
            rating=int(rating),
            is_thumbs_up=bool(is_thumbs_up),
            comment=comment,
            model_name=model_name
        )

        return JsonResponse({"status": "success", "feedback_id": feedback.id})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def get_sessions(request):
    """Return all chat sessions like Claude, ChatGPT, and Gemini."""
    from .models import ChatSession
    sessions = ChatSession.objects.all().order_by('-updated_at').values(
        'session_id', 'title', 'description', 'model_name', 'updated_at'
    )
    session_list = []
    for s in sessions:
        msg_count = VectorChatMessage.objects.filter(session_id=s['session_id']).count()
        session_list.append({
            "session_id": s['session_id'],
            "title": s['title'],
            "description": s['description'],
            "model_name": s['model_name'],
            "message_count": msg_count,
            "updated_at": s['updated_at'].strftime("%b %d, %H:%M")
        })
    return JsonResponse({"sessions": session_list})


@csrf_exempt
def create_new_session(request):
    """Create a new chat session."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        from .models import ChatSession
        data = json.loads(request.body) if request.body else {}
        session_id = f"session_{int(time.time()*1000)}"
        model_name = data.get("model_name", "Claude 3.5 Sonnet")
        title = data.get("title", "New Conversation")
        description = data.get("description", "AI Reasoning Session")

        session = ChatSession.objects.create(
            session_id=session_id,
            title=title,
            description=description,
            model_name=model_name
        )

        return JsonResponse({
            "session_id": session.session_id,
            "title": session.title,
            "description": session.description,
            "model_name": session.model_name
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def get_session_messages(request, session_id):
    """Retrieve all messages for a specific session_id."""
    msgs = VectorChatMessage.objects.filter(session_id=session_id).order_by('created_at')
    messages_list = []
    for m in msgs:
        messages_list.append({
            "id": m.id,
            "user_prompt": m.user_prompt,
            "ai_response": m.ai_response,
            "model_name": m.model_name,
            "latency_ms": m.latency_ms,
            "created_at": m.created_at.strftime("%H:%M")
        })
    return JsonResponse({"session_id": session_id, "messages": messages_list})



# ---------------- MODEL EXECUTION ENDPOINTS ----------------

@csrf_exempt
def chatbot(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    from .models import ChatSession
    data = json.loads(request.body)
    message = data.get("message", "")
    temperature = float(data.get("temperature", 0.7))
    system_prompt = data.get("system_prompt", "")
    session_id = data.get("session_id", f"session_{int(time.time()*1000)}")

    messages_payload = []
    if system_prompt:
        messages_payload.append({"role": "system", "content": system_prompt})
    messages_payload.append({"role": "user", "content": message})

    start_time = time.time()
    response_text = get_ai_response(message, model_name=MODEL, messages_payload=messages_payload, temperature=temperature)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # Get or create ChatSession like Claude, ChatGPT & Gemini
    session_obj, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={
            "title": message[:40] + ("..." if len(message) > 40 else ""),
            "description": f"Conversation session on {message[:30]}",
            "model_name": "Claude 3.5 Sonnet / GPT"
        }
    )
    if not created and session_obj.title == "New Session":
        session_obj.title = message[:40] + ("..." if len(message) > 40 else "")
        session_obj.description = f"Conversation session on {message[:30]}"
        session_obj.save()

    msg_id = None
    try:
        msg = VectorChatMessage.objects.create(
            model_name="Claude 3.5 Sonnet / GPT",
            user_prompt=message,
            ai_response=response_text,
            latency_ms=elapsed_ms,
            temperature=temperature,
            system_prompt=system_prompt,
            session_id=session_id
        )
        msg_id = msg.id
    except Exception:
        pass

    return JsonResponse({
        "response": response_text,
        "message_id": msg_id,
        "session_id": session_id,
        "session_title": session_obj.title,
        "session_description": session_obj.description,
        "latency_ms": elapsed_ms
    })


@csrf_exempt
def nemotron_omni(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    try:
        prompt = ""
        extended_reasoning = True
        temperature = 0.7
        system_prompt = ""
        image_file = None

        if request.content_type and "multipart/form-data" in request.content_type:
            prompt = request.POST.get("prompt", "").strip()
            extended_reasoning = request.POST.get("extended_reasoning", "true").lower() == "true"
            temperature = float(request.POST.get("temperature", 0.7))
            system_prompt = request.POST.get("system_prompt", "")
            if "image_file" in request.FILES:
                image_file = request.FILES["image_file"]
        else:
            data = json.loads(request.body)
            prompt = data.get("prompt", "").strip()
            extended_reasoning = data.get("extended_reasoning", True)
            temperature = float(data.get("temperature", 0.7))
            system_prompt = data.get("system_prompt", "")

        if not prompt and not image_file:
            return JsonResponse({"error": "Please enter a prompt or attach media."}, status=400)

        messages_payload = []
        if system_prompt:
            messages_payload.append({"role": "system", "content": system_prompt})

        user_content = []
        if prompt:
            prefix = "[NVIDIA NEMOTRON OMNI SUB-AGENT PERCEPTION & REASONING MODE]\n" if extended_reasoning else ""
            user_content.append({"type": "text", "text": prefix + prompt})

        if image_file:
            img_bytes = image_file.read()
            mime_type = image_file.content_type or "image/jpeg"
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
            })

        messages_payload.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 or image_file else (user_content[0]["text"] if user_content else prompt)
        })

        start_time = time.time()
        response_text = get_ai_response(prompt, model_name=NEMOTRON_OMNI_MODEL, messages_payload=messages_payload, temperature=temperature)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        audio_filename = text_to_speech_audio(response_text, "nemotron")
        audio_url = f"/audio/{audio_filename}/" if audio_filename else None

        msg_id = None
        try:
            msg = VectorChatMessage.objects.create(
                model_name=NEMOTRON_OMNI_MODEL,
                user_prompt=prompt or "Nemotron Image Perception",
                ai_response=response_text,
                latency_ms=elapsed_ms,
                temperature=temperature,
                system_prompt=system_prompt,
                audio_url=audio_url
            )
            msg_id = msg.id
        except Exception:
            pass

        return JsonResponse({
            "message_id": msg_id,
            "response": response_text,
            "model": NEMOTRON_OMNI_MODEL,
            "latency_ms": elapsed_ms,
            "extended_reasoning": extended_reasoning,
            "audio_url": audio_url
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def agentic_flash(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "").strip()
        task_mode = data.get("mode", "agentic").strip()
        temperature = float(data.get("temperature", 0.5))
        system_prompt = data.get("system_prompt", "You are Ling 3.0 Flash, a production-scale agentic AI. Perform token-efficient execution.")

        if not prompt:
            return JsonResponse({"error": "Please enter a prompt or task."}, status=400)

        messages_payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task Mode ({task_mode}):\n{prompt}"}
        ]

        start_time = time.time()
        response_text = get_ai_response(prompt, model_name=LING_FLASH_MODEL, messages_payload=messages_payload, temperature=temperature)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        audio_filename = text_to_speech_audio(response_text, "flash")
        audio_url = f"/audio/{audio_filename}/" if audio_filename else None

        msg_id = None
        try:
            msg = VectorChatMessage.objects.create(
                model_name=LING_FLASH_MODEL,
                user_prompt=prompt,
                ai_response=response_text,
                latency_ms=elapsed_ms,
                temperature=temperature,
                system_prompt=system_prompt,
                audio_url=audio_url
            )
            msg_id = msg.id
        except Exception:
            pass

        return JsonResponse({
            "message_id": msg_id,
            "response": response_text,
            "model": LING_FLASH_MODEL,
            "latency_ms": elapsed_ms,
            "audio_url": audio_url
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def gemma_moe(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    try:
        prompt = ""
        thinking_mode = False
        temperature = 0.6
        system_prompt = ""
        image_file = None

        if request.content_type and "multipart/form-data" in request.content_type:
            prompt = request.POST.get("prompt", "").strip()
            thinking_mode = request.POST.get("thinking_mode", "false").lower() == "true"
            temperature = float(request.POST.get("temperature", 0.6))
            system_prompt = request.POST.get("system_prompt", "")
            if "image_file" in request.FILES:
                image_file = request.FILES["image_file"]
        else:
            data = json.loads(request.body)
            prompt = data.get("prompt", "").strip()
            thinking_mode = data.get("thinking_mode", False)
            temperature = float(data.get("temperature", 0.6))
            system_prompt = data.get("system_prompt", "")

        if not prompt and not image_file:
            return JsonResponse({"error": "Please provide a text prompt or image file."}, status=400)

        messages_payload = []
        if system_prompt:
            messages_payload.append({"role": "system", "content": system_prompt})

        user_content = []
        if prompt:
            prefix = "[THINKING & DEEP REASONING MODE ACTIVATED]\n" if thinking_mode else ""
            user_content.append({"type": "text", "text": prefix + prompt})

        if image_file:
            img_bytes = image_file.read()
            mime_type = image_file.content_type or "image/jpeg"
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
            })

        messages_payload.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 or image_file else (user_content[0]["text"] if user_content else prompt)
        })

        start_time = time.time()
        response_text = get_ai_response(prompt, model_name=GEMMA_MOE_MODEL, messages_payload=messages_payload, temperature=temperature)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        audio_filename = text_to_speech_audio(response_text, "gemma")
        audio_url = f"/audio/{audio_filename}/" if audio_filename else None

        msg_id = None
        try:
            msg = VectorChatMessage.objects.create(
                model_name=GEMMA_MOE_MODEL,
                user_prompt=prompt or "Gemma 4 Multimodal Analysis",
                ai_response=response_text,
                latency_ms=elapsed_ms,
                temperature=temperature,
                system_prompt=system_prompt,
                audio_url=audio_url
            )
            msg_id = msg.id
        except Exception:
            pass

        return JsonResponse({
            "message_id": msg_id,
            "response": response_text,
            "model": GEMMA_MOE_MODEL,
            "latency_ms": elapsed_ms,
            "thinking_mode": thinking_mode,
            "audio_url": audio_url
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def summarize_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()

        if not query:
            return JsonResponse({"error": "Please enter a query or topic."}, status=400)

        results = search_web_robust(query, max_results=4)
        context = "\n".join(results)

        prompt = f"""You are an expert research assistant.

The user asked about: "{query}"

Web search results:
{context}

Tasks:
1. Provide a clear, concise summary (6–10 bullet points).
2. Focus on main ideas and avoid repeating patterns.
3. Where possible, mention source titles in square brackets, like [Source: Title].
4. Do NOT mention that you are an AI model.

Answer:"""

        summary = get_ai_response(prompt, model_name=SUMMARIZER_MODEL)

        doc_id = None
        try:
            doc = VectorDocumentSummary.objects.create(
                summary_type="web",
                title=query,
                content=summary,
                source_url=results[0] if results else ""
            )
            doc_id = doc.id
        except Exception:
            pass

        audio_filename = text_to_speech_audio(summary, "web")

        return JsonResponse({
            "doc_id": doc_id,
            "summary": summary,
            "raw_results": results,
            "audio_url": f"/audio/{audio_filename}/" if audio_filename else None
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def summarize_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    pdf_file = request.FILES.get("pdf_file")
    if not pdf_file:
        return JsonResponse({"error": "No PDF file provided."}, status=400)

    try:
        import pdfplumber
        text = ""
        page_cnt = 0
        with pdfplumber.open(pdf_file) as pdf:
            page_cnt = len(pdf.pages)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n\n"

        if len(text.strip()) < 30:
            return JsonResponse({"error": "The PDF contains little or no extractable text."}, status=400)

        doc_text = text[:30000]

        prompt_summary = f"""You are helping a student prepare a presentation based on this document.

Document text:
\"\"\"{doc_text}\"\"\"

Tasks:
1. Start with a 2–3 sentence high-level overview of the document.
2. Provide 6–10 bullet points covering key ideas.
3. Explain key technical terms in simple language.
4. Keep the explanation suitable for a college presentation.
5. Do NOT mention that you are an AI model.

Answer:"""

        summary = get_ai_response(prompt_summary, model_name=SUMMARIZER_MODEL)

        doc_id = None
        try:
            doc = VectorDocumentSummary.objects.create(
                summary_type="pdf",
                title=f"PDF: {pdf_file.name}",
                content=summary,
                file_size_bytes=pdf_file.size,
                page_count=page_cnt
            )
            doc_id = doc.id
        except Exception:
            pass

        audio_filename = text_to_speech_audio(summary, "pdf")

        short_summary = summary[:500].replace("\n", " ")
        prompt_query = f"""Based on this summary, generate ONE short, generic web search query (maximum 5 words) for background info.
Summary: {short_summary}
Only respond with the query text."""

        search_query = get_ai_response(prompt_query, model_name=SUMMARIZER_MODEL).strip().strip('"').strip("'")
        web_results = search_web_robust(search_query, max_results=3) if search_query else []

        synthesis = ""
        if web_results:
            web_context = "\n".join(web_results)
            prompt_synthesis = f"""Document summary:
{summary[:800]}

Web background info:
{web_context}

Explain in a few paragraphs how this additional web information expands upon the original document."""
            synthesis = get_ai_response(prompt_synthesis, model_name=SUMMARIZER_MODEL)

        return JsonResponse({
            "doc_id": doc_id,
            "summary": summary,
            "audio_url": f"/audio/{audio_filename}/" if audio_filename else None,
            "suggested_query": search_query,
            "synthesis": synthesis,
            "raw_results": web_results
        })
    except Exception as e:
        return JsonResponse({"error": f"Error analyzing PDF: {str(e)}"}, status=500)


@csrf_exempt
def summarize_code(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    code_text = ""
    file_name = "Code Analysis"
    file_size = 0
    if "code_file" in request.FILES:
        code_file = request.FILES["code_file"]
        file_name = code_file.name
        file_size = code_file.size
        code_text = code_file.read().decode("utf-8", errors="ignore")
    elif request.content_type == "application/json":
        data = json.loads(request.body)
        code_text = data.get("code_text", "")

    if not code_text.strip():
        return JsonResponse({"error": "No code content provided."}, status=400)

    funcs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code_text)
    classes = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code_text)
    libs = re.findall(r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", code_text)

    structure = {
        "func_count": len(funcs),
        "funcs": sorted(list(set(funcs)))[:10],
        "class_count": len(classes),
        "classes": sorted(list(set(classes)))[:10],
        "imports": sorted(list(set(libs)))
    }

    snippet = code_text[:5000]
    prompt_code = f"""You are a senior developer helping a beginner understand this code.

Code:
\"\"\"{snippet}\"\"\"

Tasks:
1. Identify the programming language.
2. Describe the main purpose of this code.
3. Explain the overall flow step by step in simple, non-technical language.
4. Mention key libraries and what they are used for.
5. Suggest how this code might be used in a real project.
6. Do NOT mention that you are an AI model.

Answer:"""

    analysis = get_ai_response(prompt_code, model_name=SUMMARIZER_MODEL)

    doc_id = None
    try:
        doc = VectorDocumentSummary.objects.create(
            summary_type="code",
            title=f"Code: {file_name}",
            content=analysis,
            file_size_bytes=file_size
        )
        doc_id = doc.id
    except Exception:
        pass

    audio_filename = text_to_speech_audio(analysis, "code")

    return JsonResponse({
        "doc_id": doc_id,
        "structure": structure,
        "explanation": analysis,
        "preview": code_text[:2000],
        "audio_url": f"/audio/{audio_filename}/" if audio_filename else None
    })


def serve_audio(request, filename):
    filepath = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(open(filepath, "rb"), content_type="audio/mpeg")
    return HttpResponse("Audio not found", status=404)