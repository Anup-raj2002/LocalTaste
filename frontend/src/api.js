const BASE_URL = "http://127.0.0.1:8000/api";

export async function parseAndRecommend(text) {
  const res = await fetch(`${BASE_URL}/parse-and-recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function classifySentiment(text) {
  const res = await fetch(`${BASE_URL}/sentiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}
