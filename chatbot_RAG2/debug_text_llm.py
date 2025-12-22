# 将PICO里的字段补充道pico里

import json

time curl http://192.168.156.110:1025/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer -"   -d '{
    "model": "Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {
        "role": "user",
        "content": "介绍一下北京"
      }
    ],
    "chat_template_kwargs": {
        "enable_thinking": false
    },
    "max_tokens":512,
    "presence_penalty": 1.03,
    "frequency_penalty": 1.0,
    "seed": null,
    "temperature": 0.5,
    "top_p": 0.95,
    "stream": false
  }'


time curl http://10.6.52.31:1025/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer -"   -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {
        "role": "user",
        "content": "介绍一下北京"
      }
    ],
    "max_tokens":512,
    "presence_penalty": 1.03,
    "frequency_penalty": 1.0,
    "seed": null,
    "temperature": 0.5,
    "top_p": 0.95,
    "stream": false
  }'
