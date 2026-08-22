"""
Recipe Chain — учебная цепочка из 4 LLM-запросов для генерации рецепта.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
OUTPUT_PATH = EXAMPLES_DIR / "sample_output.md"
CHAIN_REPORT_PATH = EXAMPLES_DIR / "chain_steps.md"


def build_analysis_chain(llm: ChatOpenAI):
    """Этап 1: анализ пользовательского запроса."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — кулинарный аналитик. Разбирай запросы пользователей на структурированный "
                "анализ. Отвечай только на русском языке. Будь конкретным и практичным.",
            ),
            (
                "human",
                "Проанализируй запрос пользователя на рецепт.\n\n"
                "Запрос: {user_request}\n\n"
                "В ответе укажи:\n"
                "1. Желаемое блюдо или тип блюда\n"
                "2. Доступные ингредиенты\n"
                "3. Ограничения (время, оборудование, диета, аллергии и т.д.)\n"
                "4. Уровень сложности и порции (если можно вывести)\n"
                "5. Ключевые ожидания пользователя",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_strategy_chain(llm: ChatOpenAI):
    """Этап 2: подбор стратегии приготовления."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — шеф-повар и методист. На основе анализа запроса предлагай стратегию "
                "приготовления. Отвечай только на русском языке.",
            ),
            (
                "human",
                "На основе анализа запроса предложи стратегию приготовления.\n\n"
                "Анализ запроса:\n{analysis}\n\n"
                "В ответе укажи:\n"
                "1. Подходящий тип блюда и техника (жарка, тушение, салат и т.д.)\n"
                "2. Последовательность основных этапов (кратко)\n"
                "3. Оборудование и посуда\n"
                "4. Критичные моменты (время, температура, безопасность)\n"
                "5. Как уложиться в ограничения пользователя",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_recipe_chain(llm: ChatOpenAI):
    """Этап 3: генерация черновика рецепта."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — автор кулинарных рецептов. Пиши понятные рецепты для домашней кухни. "
                "Отвечай только на русском языке. Соблюдай анализ и стратегию.",
            ),
            (
                "human",
                "Составь полный рецепт на основе анализа и стратегии.\n\n"
                "Анализ запроса:\n{analysis}\n\n"
                "Стратегия приготовления:\n{strategy}\n\n"
                "Оформи рецепт со следующими разделами (используй эти заголовки):\n"
                "## Название блюда\n"
                "## Краткое описание\n"
                "## Ингредиенты\n"
                "## Пошаговое приготовление\n"
                "## Примерное время\n"
                "## Советы по замене ингредиентов\n\n"
                "Рецепт должен быть реалистичным, безопасным и соответствовать ограничениям.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_review_chain(llm: ChatOpenAI):
    """Этап 4: проверка и улучшение рецепта."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — редактор и эксперт по пищевой безопасности. Проверяй рецепты, "
                "исправляй ошибки и улучшай формулировки. Отвечай только на русском языке.",
            ),
            (
                "human",
                "Проверь и улучши черновик рецепта. Сохрани все разделы, исправь неточности.\n\n"
                "Исходный запрос пользователя:\n{user_request}\n\n"
                "Черновик рецепта:\n{recipe_draft}\n\n"
                "Сверь рецепт с исходным запросом: ингредиенты, ограничения (время, "
                "оборудование, диета) и ожидания должны соблюдаться.\n\n"
                "В финальной версии обязательно должны быть разделы:\n"
                "- Название блюда\n"
                "- Краткое описание\n"
                "- Список ингредиентов\n"
                "- Пошаговое приготовление\n"
                "- Примерное время\n"
                "- Советы по замене ингредиентов\n"
                "- Раздел «Проверка качества» — кратко укажи, что рецепт логичен, "
                "безопасен и соответствует ограничениям пользователя\n\n"
                "Верни только улучшенный рецепт в формате Markdown.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def create_llm() -> ChatOpenAI:
    """Создаёт LLM с параметрами из переменных окружения."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "Ошибка: не задан OPENAI_API_KEY.\n"
            "Скопируйте .env.example в .env и укажите ключ API OpenAI."
        )
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def print_stage_result(title: str, content: str) -> None:
    """Выводит результат этапа цепочки в терминал."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")
    print(content)


def run_chain(user_request: str) -> tuple[str, str, str, str]:
    """Последовательно выполняет все 4 этапа цепочки."""
    llm = create_llm()

    analysis_chain = build_analysis_chain(llm)
    strategy_chain = build_strategy_chain(llm)
    recipe_chain = build_recipe_chain(llm)
    review_chain = build_review_chain(llm)

    print("[1/4] Анализ запроса...")
    analysis = analysis_chain.invoke({"user_request": user_request})
    print_stage_result("ЭТАП 1/4 — АНАЛИЗ ЗАПРОСА", analysis)

    print("[2/4] Подбор стратегии...")
    strategy = strategy_chain.invoke({"analysis": analysis})
    print_stage_result("ЭТАП 2/4 — СТРАТЕГИЯ ПРИГОТОВЛЕНИЯ", strategy)

    print("[3/4] Генерация рецепта...")
    recipe_draft = recipe_chain.invoke({"analysis": analysis, "strategy": strategy})
    print_stage_result("ЭТАП 3/4 — ЧЕРНОВИК РЕЦЕПТА", recipe_draft)

    print("[4/4] Проверка результата...")
    final_recipe = review_chain.invoke(
        {"user_request": user_request, "recipe_draft": recipe_draft}
    )
    print_stage_result("ЭТАП 4/4 — ФИНАЛЬНАЯ ВЕРСИЯ ПОСЛЕ РЕВЬЮ", final_recipe)

    return analysis, strategy, recipe_draft, final_recipe


def ensure_examples_dir() -> None:
    """Создаёт папку examples, если её нет."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def save_output(content: str) -> None:
    """Сохраняет финальный рецепт в examples/sample_output.md."""
    ensure_examples_dir()
    header = (
        "# Пример результата Recipe Chain\n\n"
        "> Сгенерировано автоматически скриптом `recipe_chain.py`\n\n"
        "---\n\n"
    )
    OUTPUT_PATH.write_text(header + content, encoding="utf-8")


def save_chain_report(
    user_request: str,
    analysis: str,
    strategy: str,
    recipe_draft: str,
    final_recipe: str,
) -> None:
    """Сохраняет полный отчёт по всем этапам в examples/chain_steps.md."""
    ensure_examples_dir()
    report = (
        "# Recipe Chain — отчёт по шагам\n\n"
        "## Исходный запрос\n\n"
        f"{user_request}\n\n"
        "## Этап 1. Анализ запроса\n\n"
        f"{analysis}\n\n"
        "## Этап 2. Стратегия приготовления\n\n"
        f"{strategy}\n\n"
        "## Этап 3. Черновик рецепта\n\n"
        f"{recipe_draft}\n\n"
        "## Этап 4. Финальная версия после ревью\n\n"
        f"{final_recipe}\n"
    )
    CHAIN_REPORT_PATH.write_text(report, encoding="utf-8")


def print_usage() -> None:
    """Показывает пример запуска."""
    print("Использование:")
    print('  python recipe_chain.py "Ваш запрос на рецепт"')
    print()
    print("Пример:")
    print(
        '  python recipe_chain.py "Хочу быстрый ужин из курицы, '
        'риса и овощей, без духовки, до 30 минут"'
    )


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    user_request = " ".join(sys.argv[1:]).strip()
    if not user_request:
        print_usage()
        sys.exit(1)

    print("Recipe Chain — генерация рецепта\n")
    print(f"Запрос: {user_request}\n")

    try:
        analysis, strategy, recipe_draft, final_recipe = run_chain(user_request)
    except Exception as exc:
        print(f"\nОшибка при выполнении цепочки: {exc}")
        sys.exit(1)

    save_output(final_recipe)
    save_chain_report(user_request, analysis, strategy, recipe_draft, final_recipe)
    print(f"\nФинальный рецепт сохранён в: {OUTPUT_PATH}")
    print(f"Отчёт по этапам сохранён в: {CHAIN_REPORT_PATH}")


if __name__ == "__main__":
    main()
