from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);

