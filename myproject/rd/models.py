from django.db import models

try:
    from pgvector.django import VectorField
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


class AIModelConfig(models.Model):
    """
    Stores AI model configurations, providers, and capabilities.
    """
    model_id = models.CharField(max_length=150, unique=True)
    display_name = models.CharField(max_length=150)
    provider = models.CharField(max_length=100, default="OpenRouter")
    category = models.CharField(max_length=50, default="General")  # 'omni', 'moe', 'flash', 'summarizer', 'chat'
    description = models.TextField(blank=True, default="")
    context_window = models.CharField(max_length=50, default="128K")
    default_temperature = models.FloatField(default=0.7)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.display_name} ({self.model_id})"


class ChatSession(models.Model):
    """
    Stores grouped chat conversation sessions like Claude, ChatGPT, and Gemini.
    """
    session_id = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=255, default="New Session")
    description = models.TextField(blank=True, default="AI Conversation Session")
    model_name = models.CharField(max_length=150, default="Claude 3.5 Sonnet")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"[{self.model_name}] {self.title}"



class VectorChatMessage(models.Model):
    """
    Stores Chat, Nemotron, Gemma 4, and Ling Flash messages with rich metadata and pgvector support.
    """
    model_name = models.CharField(max_length=150, default="chat")
    user_prompt = models.TextField()
    ai_response = models.TextField()
    latency_ms = models.FloatField(default=0.0)
    tokens_used = models.IntegerField(default=0)
    temperature = models.FloatField(default=0.7)
    system_prompt = models.TextField(blank=True, default="")
    audio_url = models.CharField(max_length=500, blank=True, null=True)
    image_url = models.CharField(max_length=500, blank=True, null=True)
    tags = models.CharField(max_length=255, blank=True, default="")
    session_id = models.CharField(max_length=100, blank=True, default="default")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.model_name}] {self.user_prompt[:30]}..."


class VectorDocumentSummary(models.Model):
    """
    Stores PDF, Web, and Code summaries with document metadata and pgvector support.
    """
    summary_type = models.CharField(max_length=50)  # 'web', 'pdf', 'code'
    title = models.CharField(max_length=255)
    content = models.TextField()
    file_size_bytes = models.BigIntegerField(default=0)
    page_count = models.IntegerField(default=1)
    source_url = models.CharField(max_length=500, blank=True, default="")
    keywords = models.CharField(max_length=255, blank=True, default="")
    language = models.CharField(max_length=50, default="English")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.summary_type}] {self.title}"


class UserFeedback(models.Model):
    """
    Stores user ratings, thumbs up/down, and comments for AI outputs.
    """
    chat_message = models.ForeignKey(VectorChatMessage, on_delete=models.SET_NULL, null=True, blank=True, related_name="feedbacks")
    summary_document = models.ForeignKey(VectorDocumentSummary, on_delete=models.SET_NULL, null=True, blank=True, related_name="feedbacks")
    rating = models.IntegerField(default=5)  # 1 to 5 stars
    is_thumbs_up = models.BooleanField(default=True)
    comment = models.TextField(blank=True, default="")
    model_name = models.CharField(max_length=150, default="Unknown")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        status = "👍" if self.is_thumbs_up else "👎"
        return f"Feedback {status} ({self.rating}/5) for {self.model_name}"


class PresetPrompt(models.Model):
    """
    Stores template prompts for quick workflow execution.
    """
    title = models.CharField(max_length=150)
    category = models.CharField(max_length=50, default="General")  # 'Code', 'Summary', 'Reasoning', 'Creative'
    prompt_text = models.TextField()
    system_instruction = models.TextField(blank=True, default="")
    icon = models.CharField(max_length=50, default="fa-rocket")
    is_featured = models.BooleanField(default=True)

    def __str__(self):
        return f"[{self.category}] {self.title}"

