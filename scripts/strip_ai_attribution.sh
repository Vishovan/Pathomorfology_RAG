#!/usr/bin/env bash
# Скрипт защиты репозитория от появления нейросетей (Claude, Cursor, Copilot) в Contributors на GitHub.
#
# Что делает скрипт:
# 1. Устанавливает локальный git hook, удаляющий Co-authored-by от AI-агентов.
# 2. Сканирует историю коммитов на наличие упоминаний AI в соавторах.
# 3. Предоставляет команду для очистки истории, если нейросеть уже попала в коммиты.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_DEST="${ROOT_DIR}/.git/hooks/commit-msg"

echo "=== Защита от AI-контрибьюторов (Claude, Copilot, Cursor) ==="

# 1. Установка hook
cat << 'EOF' > "$HOOK_DEST"
#!/usr/bin/env bash
COMMIT_MSG_FILE="$1"
if [[ -f "$COMMIT_MSG_FILE" ]]; then
    sed -i.bak -E '/[Cc]o-[Aa]uthored-[Bb]y:[[:space:]]*.*(Claude|Anthropic|Copilot|Cursor|Assistant|OpenAI|ChatGPT|bot|AI).*/Id' "$COMMIT_MSG_FILE"
    sed -i.bak -E '/[Gg]enerated[[:space:]]+by[[:space:]]+(Claude|Cursor|Copilot|AI|ChatGPT)/Id' "$COMMIT_MSG_FILE"
    rm -f "${COMMIT_MSG_FILE}.bak"
fi
exit 0
EOF

chmod +x "$HOOK_DEST"
echo "✅ Git hook установлен в ${HOOK_DEST}"

# 2. Сканирование существующей истории
echo "🔍 Проверка существующей истории коммитов..."
MATCHES=$(git log --all --format='%H %B' | grep -iE 'co-authored-by:.*(claude|anthropic|copilot|cursor|assistant|ai)' || true)

if [[ -z "$MATCHES" ]]; then
    echo "🎉 В текущей истории коммитов НЕТ упоминаний нейросетей в соавторах. Репозиторий чист!"
else
    echo "⚠️ Найдены коммиты с AI-соавторами:"
    echo "$MATCHES"
    echo ""
    echo "Для полной очистки истории выполните команду:"
    echo "git filter-branch -f --msg-filter 'sed -E \"/[Cc]o-[Aa]uthored-[Bb]y:.*(Claude|Anthropic|Copilot|Cursor)/Id\"' HEAD"
    echo "Затем: git push origin --force"
fi

echo "=== Готово ==="
