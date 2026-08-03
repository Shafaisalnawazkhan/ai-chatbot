from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/', views.chatbot, name='chatbot'),
    path('nemotron_omni/', views.nemotron_omni, name='nemotron_omni'),
    path('gemma_moe/', views.gemma_moe, name='gemma_moe'),
    path('agentic_flash/', views.agentic_flash, name='agentic_flash'),
    path('summarize_text/', views.summarize_text, name='summarize_text'),
    path('summarize_pdf/', views.summarize_pdf, name='summarize_pdf'),
    path('summarize_code/', views.summarize_code, name='summarize_code'),
    path('audio/<str:filename>/', views.serve_audio, name='serve_audio'),
    path('ai-logo/', views.ai_logo, name='ai_logo'),
    # Enhanced API Routes
    path('api/models/', views.get_models_list, name='get_models_list'),
    path('api/prompts/', views.get_preset_prompts, name='get_preset_prompts'),
    path('api/analytics/', views.get_analytics, name='get_analytics'),
    path('api/history/', views.get_history, name='get_history'),
    path('api/feedback/', views.submit_feedback, name='submit_feedback'),
    path('api/sessions/', views.get_sessions, name='get_sessions'),
    path('api/sessions/new/', views.create_new_session, name='create_new_session'),
    path('api/sessions/<str:session_id>/', views.get_session_messages, name='get_session_messages'),
]