import { useState } from "react";
import { classifySentiment } from "../api";

function TasteTestTab() {
  const [reviewText, setReviewText] = useState("");
  const [sentimentResult, setSentimentResult] = useState(null);
  const [loadingSentiment, setLoadingSentiment] = useState(false);

  async function handleSentiment() {
    if (!reviewText.trim()) return;
    setLoadingSentiment(true);
    try {
      const label = await classifySentiment(reviewText);
      setSentimentResult(label);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingSentiment(false);
    }
  }

  return (
    <section className="bg-clay rounded-lg p-6 border border-cream/10">
      <h2 className="font-display text-xl text-turmeric mb-1">Taste test a review</h2>
      <p className="text-cream/50 text-xs font-mono uppercase tracking-widest mb-4">
        Our kitchen critic reads it back
      </p>
      <textarea
        value={reviewText}
        onChange={(e) => setReviewText(e.target.value)}
        placeholder="Paste a restaurant review here..."
        rows={4}
        className="w-full bg-ink/40 border border-cream/15 rounded-md px-3 py-2 text-sm text-cream placeholder-cream/30 focus:outline-none focus:border-turmeric resize-none"
      />
      <button
        onClick={handleSentiment}
        disabled={!reviewText.trim() || loadingSentiment}
        className="mt-3 bg-turmeric hover:bg-turmeric/90 disabled:bg-cream/10 disabled:cursor-not-allowed text-ink font-mono text-xs uppercase tracking-widest font-semibold px-5 py-2.5 rounded transition"
      >
        {loadingSentiment ? "Tasting..." : "Taste It"}
      </button>

      {sentimentResult && (
        <div className="mt-5 flex justify-center">
          <span
            className={`stamp font-mono font-bold text-sm uppercase tracking-widest border-4 rounded px-4 py-2 ${
              sentimentResult === "positive" ? "border-curry text-curry" : "border-chili text-chili"
            }`}
          >
            {sentimentResult === "positive" ? "✓ Approved" : "✕ Sent Back"}
          </span>
        </div>
      )}
    </section>
  );
}

export default TasteTestTab;