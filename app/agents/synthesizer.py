import os
import json
import re
from groq import Groq
from openai import OpenAI
from typing import List, Dict
from app.api.schemas import FinalSynthesis

class SynthesizerAgent:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        # МАГИЯ ЖОНГЛИРОВАНИЯ v3.0: Повышенная выживаемость и корректные эндпоинты
        self.models_pipeline = [
            # 1. Сверх-разум (Новый эндпоинт): DeepSeek R1
            {
                "client_type": "openrouter",
                "model": "deepseek/deepseek-r1", # Без :free часто работает стабильнее через балансировщик
                "max_chars": 40000,
                "chunk_size": 6000
            },
            # 2. Король контекста: Gemini 2.0 Flash (OpenRouter / Google)
            # У Gemini лучшие лимиты на бесплатные запросы и 1M+ контекста
            {
                "client_type": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "max_chars": 60000, 
                "chunk_size": 10000
            },
            # 3. Основной танк (Groq): Llama 3.3 70B
            {
                "client_type": "groq",
                "model": "llama-3.3-70b-versatile",
                "max_chars": 28000,
                "chunk_size": 4000
            },
            # 4. Проверенный китаец: Qwen 2.5 72B Instruct
            # Исправленный эндпоинт (убираем :free, если летит 404)
            {
                "client_type": "openrouter",
                "model": "qwen/qwen-2.5-72b-instruct",
                "max_chars": 30000,
                "chunk_size": 4000
            },
            # 5. Твой спаситель: StepFun 3.5 Flash
            # Оставляем как самый надежный запасной аэродром
            {
                "client_type": "openrouter",
                "model": "stepfun/step-3.5-flash",
                "max_chars": 35000,
                "chunk_size": 5000
            },
            # 6. Стабильный европеец: Mistral Small 24B
            # Она умнее 8B моделей, но гораздо стабильнее 70B гигантов
            {
                "client_type": "openrouter",
                "model": "mistralai/mistral-small-24b-instruct-2501",
                "max_chars": 25000,
                "chunk_size": 4000
            },
            # 7. Последний рубеж (Groq): Llama 3.1 8B
            {
                "client_type": "groq",
                "model": "llama-3.1-8b-instant",
                "max_chars": 12000,
                "chunk_size": 2000
            }
        ]

    def _extract_json(self, text: str) -> str:
        """Улучшенный парсер: вырезает мысли DeepSeek и находит чистый JSON"""
        # Убираем блок <think> если он есть
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
        
        # Ищем сам JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def generate_guide(self, user_query: str, query_language: str, scraped_data: List[Dict[str, str]]) -> FinalSynthesis:
        
        system_prompt = f"""
        You are an Elite Gaming Meta Analyst and a "Gamer Bro". 
        TARGET LANGUAGE: {query_language}
        
        INSTRUCTIONS:
        1. GAMER BRO TONE (CRITICAL): Start with 1-2 sentences of friendly, conversational "bro" banter responding to their exact problem. Do not sound like a robot. 
           Example: "Ох, бро, понимаю. Отлетать по КД на Эстесе — это классика боли. Но сейчас мы пересоберем твой билд..."
        
        2. NO HALLUCINATIONS (CRITICAL): 
           - DO NOT invent fake items like "Звезда Зла" or "Палочка Света". 
           - DO NOT invent fake mechanics like "играть 30 минут ради уровня".
           - Use EXACT item names from the text. If you don't know the exact names, give general advice without naming fake items.
           
        3. BILINGUAL TERMINOLOGY: EVERY time you name a real item/skill, write it in the target language followed by its ORIGINAL ENGLISH NAME in parentheses.
           Example: Фляга Оазиса (Flask of the Oasis).
        
        4. STRUCTURE & NO-REPETITION: 
           - Use short bullet points. 
           - DO NOT output markdown tables.
           - NO summary paragraphs at the end. When you finish the tips, stop!
           
        5. OUTPUT LANGUAGE: Write the markdown_guide ENTIRELY in the TARGET LANGUAGE ({query_language}), except for the English terms.
        
        OUTPUT FORMAT:
        You MUST output ONLY a valid JSON object. Do NOT wrap it in ```json blocks. Use EXACTLY these 4 keys:
        {{
          "verified_version": "string (patch number or 'Unknown')",
          "previous_version": "string (previous patch or 'Unknown')",
          "confidence_score": 95,
          "markdown_guide": "string (your full formatted Markdown guide here)"
        }}
        """

        for config in self.models_pipeline:
            try:
                print(f"\n[*] [СИНТЕЗАТОР] Запускаю модель: {config['model']} (Лимит: {config['max_chars']} симв.)")
                
                sources_text = ""
                for idx, item in enumerate(scraped_data):
                    clean_text = item['content'][:config['chunk_size']]
                    if len(sources_text) + len(clean_text) > config['max_chars']:
                        print(f"    - Контекст заполнен ({len(sources_text)} симв.). Взято источников: {idx}")
                        break
                    sources_text += f"\n--- SOURCE {idx + 1} ({item['url']}) ---\n{clean_text}\n"

                user_message_content = f"""
                RAW SCRAPED DATA:
                {sources_text}
                
                ================================
                CRITICAL FINAL INSTRUCTION:
                The user's exact query is: "{user_query}"
                
                Start with a Gamer Bro empathetic opening. Use English names in parentheses. DO NOT HALLUCINATE ITEMS. Output JSON now.
                """

                if config["client_type"] == "groq":
                    completion = self.groq_client.chat.completions.create(
                        model=config["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message_content}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.6 
                    )
                else: 
                    completion = self.or_client.chat.completions.create(
                        model=config["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message_content}
                        ],
                        # ВНИМАНИЕ: Убрали response_format для OpenRouter, чтобы избежать 400 ошибки!
                        temperature=0.6,
                        extra_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "Gamer Meta Oracle"}
                    )
                
                # Пропускаем ответ через наш regex-парсер
                raw_response = completion.choices[0].message.content
                clean_json = self._extract_json(raw_response)
                
                validated_data = FinalSynthesis.model_validate_json(clean_json)
                print(f"[+] Успех! Модель {config['model']} справилась.")
                return validated_data

            except Exception as e:
                print(f"[!] Ошибка на модели {config['model']}: {str(e)[:150]}... Переключаюсь на план Б!")
                continue

        raise Exception("Критическая ошибка: Все резервные ИИ-модели упали.")

    def continue_chat(self, new_message: str, chat_history: List[Dict[str, str]], scraped_data: List[Dict[str, str]]) -> str:
        for config in self.models_pipeline:
            try:
                sources_text = ""
                for idx, item in enumerate(scraped_data):
                    clean_text = item['content'][:config['chunk_size']]
                    if len(sources_text) + len(clean_text) > config['max_chars']:
                        break
                    sources_text += f"\n--- SOURCE {idx + 1} ({item['url']}) ---\n{clean_text}\n"

                system_prompt = f"""
                You are an Elite Gaming Meta Analyst and Gamer Bro assisting a user.
                Answer directly using the SOURCE DATA. Use English names in parentheses. 
                DO NOT invent items or stats. Write ENTIRELY in the language of the user's message.
                
                SOURCE DATA:
                {sources_text}
                """

                messages = [{"role": "system", "content": system_prompt}]
                for msg in chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": new_message})

                if config["client_type"] == "groq":
                    completion = self.groq_client.chat.completions.create(model=config["model"], messages=messages, temperature=0.6)
                else:
                    completion = self.or_client.chat.completions.create(model=config["model"], messages=messages, temperature=0.6)
                
                return completion.choices[0].message.content
            except Exception as e:
                continue
                
        raise Exception("Chat fallback failed.")