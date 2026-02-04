import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime

import customtkinter as ctk
import pyttsx3
import speech_recognition as sr


@dataclass
class LLMConfig:
    provider: str
    api_key: str | None
    model: str


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate_response(self, prompt: str) -> str:
        provider = self.config.provider.lower()
        if provider == "gemini":
            return self._generate_with_gemini(prompt)
        if provider == "openai":
            return self._generate_with_openai(prompt)
        return (
            "I'm Loki. I can answer questions once you configure an LLM provider. "
            "Set LOKI_LLM_PROVIDER to 'openai' or 'gemini' and provide an API key."
        )

    def _generate_with_gemini(self, prompt: str) -> str:
        if not self.config.api_key:
            return "Gemini API key not configured. Please set LOKI_GEMINI_API_KEY."
        try:
            import google.generativeai as genai
        except ImportError:
            return "Gemini SDK not installed. Please install google-generativeai."
        genai.configure(api_key=self.config.api_key)
        model = genai.GenerativeModel(self.config.model)
        response = model.generate_content(prompt)
        return response.text or "I couldn't generate a response right now."

    def _generate_with_openai(self, prompt: str) -> str:
        if not self.config.api_key:
            return "OpenAI API key not configured. Please set LOKI_OPENAI_API_KEY."
        try:
            from openai import OpenAI
        except ImportError:
            return "OpenAI SDK not installed. Please install openai."
        client = OpenAI(api_key=self.config.api_key)
        response = client.responses.create(
            model=self.config.model,
            input=prompt,
        )
        return response.output_text or "I couldn't generate a response right now."


class LokiAssistant:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Loki - Voice Assistant")
        self.root.geometry("900x600")
        self.root.minsize(820, 520)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.status_var = ctk.StringVar(value="Idle")
        self.queue: queue.Queue[str] = queue.Queue()
        self.listening = False
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 185)
        self.recognizer = sr.Recognizer()

        self.llm_client = LLMClient(self._load_llm_config())

        self._build_ui()
        self._start_queue_poller()

    def _load_llm_config(self) -> LLMConfig:
        provider = os.getenv("LOKI_LLM_PROVIDER", "openai")
        api_key = None
        model = "gpt-4o-mini"
        if provider.lower() == "openai":
            api_key = os.getenv("LOKI_OPENAI_API_KEY")
            model = os.getenv("LOKI_OPENAI_MODEL", model)
        elif provider.lower() == "gemini":
            api_key = os.getenv("LOKI_GEMINI_API_KEY")
            model = os.getenv("LOKI_GEMINI_MODEL", "gemini-1.5-pro")
        return LLMConfig(provider=provider, api_key=api_key, model=model)

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self.root, fg_color="#0b1d2a")
        header.pack(fill="x", padx=20, pady=(20, 10))

        title = ctk.CTkLabel(
            header,
            text="Loki",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#7ae7ff",
        )
        title.pack(side="left", padx=20, pady=15)

        subtitle = ctk.CTkLabel(
            header,
            text="Your desktop voice assistant",
            font=ctk.CTkFont(size=14),
            text_color="#c2d3df",
        )
        subtitle.pack(side="left", padx=10)

        status_frame = ctk.CTkFrame(self.root, fg_color="#08131b")
        status_frame.pack(fill="x", padx=20, pady=(0, 10))

        status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00f5d4",
        )
        status_label.pack(side="left", padx=20, pady=10)

        self.toggle_button = ctk.CTkButton(
            status_frame,
            text="Start Listening",
            command=self.toggle_listening,
            fg_color="#1f6aa5",
            hover_color="#144870",
        )
        self.toggle_button.pack(side="right", padx=20, pady=10)

        self.chatbox = ctk.CTkTextbox(
            self.root,
            wrap="word",
            width=760,
            height=380,
            fg_color="#0f1a24",
            text_color="#e5f4ff",
        )
        self.chatbox.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.chatbox.configure(state="disabled")

        input_frame = ctk.CTkFrame(self.root, fg_color="#0f1a24")
        input_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.manual_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type a command or question...",
            fg_color="#13212f",
        )
        self.manual_entry.pack(side="left", fill="x", expand=True, padx=(10, 10), pady=10)
        self.manual_entry.bind("<Return>", lambda event: self.process_manual_input())

        send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.process_manual_input,
            fg_color="#1f6aa5",
        )
        send_button.pack(side="right", padx=(0, 10), pady=10)

    def _append_message(self, speaker: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        formatted = f"[{timestamp}] {speaker}: {message}\n\n"
        self.chatbox.configure(state="normal")
        self.chatbox.insert("end", formatted)
        self.chatbox.configure(state="disabled")
        self.chatbox.see("end")

    def _start_queue_poller(self) -> None:
        def poll():
            while not self.queue.empty():
                message = self.queue.get_nowait()
                self._append_message("Loki", message)
            self.root.after(150, poll)

        poll()

    def toggle_listening(self) -> None:
        if self.listening:
            self.listening = False
            self.status_var.set("Idle")
            self.toggle_button.configure(text="Start Listening")
            return

        self.listening = True
        self.status_var.set("Listening...")
        self.toggle_button.configure(text="Stop Listening")
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self) -> None:
        while self.listening:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio)
                self._append_message("You", text)
                self._handle_command(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                self.queue.put("I didn't catch that. Could you repeat?")
            except Exception as exc:
                self.queue.put(f"Microphone error: {exc}")
                self.listening = False
                self.status_var.set("Microphone error")
                self.toggle_button.configure(text="Start Listening")
                break

    def process_manual_input(self) -> None:
        text = self.manual_entry.get().strip()
        if not text:
            return
        self.manual_entry.delete(0, "end")
        self._append_message("You", text)
        threading.Thread(target=self._handle_command, args=(text,), daemon=True).start()

    def _handle_command(self, text: str) -> None:
        normalized = text.lower().strip()
        if normalized.startswith("open "):
            target = normalized.replace("open ", "", 1).strip()
            response = self._handle_open(target)
            self.queue.put(response)
            self._speak(response)
            return

        if normalized.startswith("go to ") or normalized.startswith("visit "):
            target = normalized.split(" ", 2)[-1]
            response = self._open_website(target)
            self.queue.put(response)
            self._speak(response)
            return

        if any(keyword in normalized for keyword in ["bye", "goodbye", "exit"]):
            response = "Goodbye! Loki is going idle."
            self.queue.put(response)
            self._speak(response)
            self.listening = False
            self.status_var.set("Idle")
            self.toggle_button.configure(text="Start Listening")
            return

        self.status_var.set("Thinking...")
        response = self.llm_client.generate_response(self._build_prompt(text))
        self.queue.put(response)
        self._speak(response)
        self.status_var.set("Listening..." if self.listening else "Idle")

    def _build_prompt(self, user_input: str) -> str:
        return (
            "You are Loki, a helpful desktop voice assistant with an intelligent, friendly personality "
            "similar to Gemini AI. Provide concise, actionable responses."
            f"\nUser: {user_input}\nLoki:"
        )

    def _handle_open(self, target: str) -> str:
        if target.startswith("http") or "." in target:
            return self._open_website(target)

        app_map = {
            "windows": {
                "chrome": "chrome",
                "notepad": "notepad",
                "calculator": "calc",
            },
            "darwin": {
                "chrome": "Google Chrome",
                "notes": "Notes",
                "calculator": "Calculator",
            },
            "linux": {
                "chrome": "google-chrome",
                "firefox": "firefox",
                "calculator": "gnome-calculator",
            },
        }
        platform_key = self._platform_key()
        command = app_map.get(platform_key, {}).get(target)
        if not command:
            return f"I don't have a shortcut for '{target}'. Try saying 'open https://...'."

        try:
            if platform_key == "darwin":
                subprocess.Popen(["open", "-a", command])
            elif platform_key == "windows":
                subprocess.Popen([command], shell=True)
            else:
                subprocess.Popen([command])
            return f"Opening {target}."
        except Exception as exc:
            return f"I couldn't open {target}: {exc}"

    def _open_website(self, target: str) -> str:
        if not target.startswith("http"):
            target = f"https://{target}"
        try:
            webbrowser.open(target)
            return f"Opening {target}."
        except Exception as exc:
            return f"I couldn't open that website: {exc}"

    def _speak(self, text: str) -> None:
        def run():
            self.status_var.set("Speaking...")
            self.engine.say(text)
            self.engine.runAndWait()
            self.status_var.set("Listening..." if self.listening else "Idle")

        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _platform_key() -> str:
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform == "darwin":
            return "darwin"
        return "linux"


def main() -> None:
    root = ctk.CTk()
    app = LokiAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
