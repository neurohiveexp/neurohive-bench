# Manual answers

For chats without an API: save each answer as `responses/<label>/<task_id>.txt` (or `<task_id>__run2.txt` for repeated runs), then run:

```bash
python -m bench score responses/<label> --model "<exact version shown in the chat>"
```

Copy the answer text exactly as the chat gave it, without edits.
