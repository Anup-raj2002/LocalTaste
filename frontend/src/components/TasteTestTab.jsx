import { useState } from "react";
import { classifySentiment } from "../api";

function TasteTestTab() {
  const [reviewText, setReviewText] = useState("");
  const [sentimentData, setSentimentData] = useState(null);
  const [loadingSentiment, setLoadingSentiment] = useState(false);

  async function handleSentiment() {
    if (!reviewText.trim()) return;
    setLoadingSentiment(true);
    try {
      const data = await classifySentiment(reviewText);
      setSentimentData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingSentiment(false);
    }
  }

  const isPositive = sentimentData?.label === "positive";

  return (
    <section className="bg-clay rounded-lg p-6 border border-cream/10">
      <h2 className="font-display text-xl text-turmeric mb-1">Customer Feedback</h2>
      <p className="text-cream/50 text-xs font-mono uppercase tracking-widest mb-4">
        Share your dining experience with us
      </p>

      <textarea
        value={reviewText}
        onChange={(e) => setReviewText(e.target.value)}
        placeholder="Write your feedback or review here..."
        rows={4}
        className="w-full bg-ink/40 border border-cream/15 rounded-md px-3 py-2 text-sm text-cream placeholder-cream/30 focus:outline-none focus:border-turmeric resize-none"
      />

      <button
        onClick={handleSentiment}
        disabled={!reviewText.trim() || loadingSentiment}
        className="mt-3 bg-turmeric hover:bg-turmeric/90 disabled:bg-cream/10 disabled:cursor-not-allowed text-ink font-mono text-xs uppercase tracking-widest font-semibold px-5 py-2.5 rounded transition"
      >
        {loadingSentiment ? "Submitting..." : "Submit Feedback"}
      </button>

      {sentimentData && sentimentData.label && sentimentData.label !== "unknown" && (
        <div className="mt-6 space-y-4 border-t border-cream/10 pt-5">
          {/* Positive or Negative Review Classification */}
          <div className="flex justify-center">
            <span
              className={`font-mono font-bold text-sm uppercase tracking-widest px-4 py-1.5 rounded border-2 ${
                isPositive
                  ? "border-curry text-curry bg-curry/10"
                  : "border-chili text-chili bg-chili/10"
              }`}
            >
              {isPositive ? "Positive Review" : "Negative Review"}
            </span>
          </div>

          {/* Appreciation or Apology Message from Gemini */}
          {sentimentData.message && (
            <div
              className={`rounded-md p-4 border-l-4 ${
                isPositive
                  ? "bg-curry/10 border-curry text-cream"
                  : "bg-chili/10 border-chili text-cream"
              }`}
            >
              <p className="font-display italic text-base leading-relaxed">
                "{sentimentData.message}"
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export default TasteTestTab;
