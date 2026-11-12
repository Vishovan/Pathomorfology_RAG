#!/usr/bin/env python
"""Бенчмарк оценки качества ответов Pathology RAG агента по патологии и ГОСТам."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathrag.agent import ask_agent

BENCHMARK_CASES = [
    {
        "id": "brca_her2_equivocal",
        "category": "Breast Cancer IHC",
        "query": "Какие критерии оценки HER2 2+ при раке молочной железы и когда требуется подтверждающее тестирование FISH?",
        "expected_intent": "breast_cancer_ihc",
        "must_contain": ["2+", "fish", "неопределен", "мембран", "10%"]
    },
    {
        "id": "brca_ki67_subtypes",
        "category": "Breast Cancer IHC",
        "query": "Каковы пороговые значения Ki-67 для дифференциации люминального А и люминального В подтипов РМЖ?",
        "expected_intent": "breast_cancer_ihc",
        "must_contain": ["ki-67", "люминальн", "20%", "hot spot"]
    },
    {
        "id": "legal_gost_registration",
        "category": "Legal & GOST",
        "query": "В чем разница между регистрацией в Роспатенте, реестром Минцифры и получением РУ в Росздравнадзоре как медизделия?",
        "expected_intent": "legal_gost",
        "must_contain": ["роспатент", "минцифры", "росздравнадзор", "эвм", "реестр"]
    },
    {
        "id": "legal_gost_standards",
        "category": "Legal & GOST",
        "query": "Какие стандарты ГОСТ регламентируют разработку Технического задания и безопасную разработку медицинского ПО?",
        "expected_intent": "legal_gost",
        "must_contain": ["34.602", "56939"]
    },
    {
        "id": "general_pneumonia_stages",
        "category": "General Pathology",
        "query": "Опишите микроскопическую картину серого опеченения при крупозной пневмонии",
        "expected_intent": "pathology_qa",
        "must_contain": ["серого опеченения", "фибрин", "нейтрофил", "альвеол"]
    },
    {
        "id": "image_search_wsi",
        "category": "Image Search",
        "query": "Покажи микропрепарат серого опеченения при крупозной пневмонии",
        "expected_intent": "image_search",
        "must_contain": ["серого опеченения", "wsi"]
    }
]


def run_benchmark():
    print("=" * 75)
    print("🧪 ЗАПУСК БЕНЧМАРКА PATHOLOGY RAG (ОЦЕНКА КАЧЕСТВА)")
    print("=" * 75)

    passed = 0
    total = len(BENCHMARK_CASES)

    for i, test in enumerate(BENCHMARK_CASES, 1):
        print(f"\n[Тест {i}/{total}] [{test['category']}] {test['id']}")
        print(f"Вопрос: {test['query']}")

        t0 = time.time()
        res = ask_agent(test["query"])
        elapsed = time.time() - t0

        intent_ok = res.get("intent") == test["expected_intent"]
        answer_text = (res.get("final_answer", "") + " " + str(res.get("retrieved_images", ""))).lower()

        missing_keywords = [kw for kw in test["must_contain"] if kw.lower() not in answer_text]
        content_ok = len(missing_keywords) == 0

        status = "PASSED" if (intent_ok and content_ok) else "FAILED"
        if status == "PASSED":
            passed += 1

        print(f"Статус: {status} (время: {elapsed:.2f}с)")
        print(f"  Маршрут графа: {res.get('intent')} (ожидался: {test['expected_intent']}) -> {'OK' if intent_ok else 'FAIL'}")
        print(f"  Покрытие терминов: {'OK' if content_ok else f'Отсутствуют: {missing_keywords}'}")
        print(f"  Цитаты: {len(res.get('citations', []))} источников")
        print(f"  Дисклеймер: {'Присутствует' if res.get('disclaimer') else 'ОТСУТСТВУЕТ'}")

    print("\n" + "=" * 75)
    print(f"ИТОГ: Успешно пройдено {passed} из {total} тестов ({passed/total*100:.1f}%)")
    print("=" * 75)

    return passed == total


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
