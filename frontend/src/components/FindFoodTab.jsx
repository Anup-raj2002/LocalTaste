import { useState } from "react";
import { parseAndRecommend } from "../api";
import RecTicket from "./RecTicket";

function FindFoodTab() {
  const [query, setQuery] = useState("");
  const [parsed, setParsed] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setRecommendations([]);
    setParsed(null);
    try {
      const data = await parseAndRecommend(query);
      if (data.error) {
        setError(data.error);
      } else {
        setParsed(data.parsed);
        setRecommendations(data.recommendations);
      }
    } catch (err) {
      setError("Something went wrong reaching the kitchen. Check the backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <section className="bg-cream torn-top rounded-b-md px-6 pb-6 shadow-xl mb-10">
        <p className="font-mono text-[0.65rem] uppercase tracking-widest text-ink/50 mb-1">
          Table for one &middot; Order taken by LocalTaste
        </p>
        <label className="block font-display text-lg text-ink font-semibold mb-2">
          Where are you, and what are you craving?
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="I'm in Koramangala craving spicy biryani... (or just 'I am in Bangalore craving South Indian & filter coffee' for city-wide top picks!)"
          rows={3}
          className="w-full bg-transparent text-ink placeholder-ink/40 border-b-2 border-dashed border-ink/25 focus:outline-none focus:border-chili text-sm py-2 resize-none font-body"
        />
        <button
          onClick={handleSubmit}
          disabled={!query.trim() || loading}
          className="mt-4 bg-chili hover:bg-chili/90 disabled:bg-ink/20 disabled:cursor-not-allowed text-cream font-mono text-xs uppercase tracking-widest font-semibold px-5 py-2.5 rounded transition"
        >
          {loading ? "Sending to kitchen..." : "Send to Kitchen"}
        </button>

        {error && <p className="mt-3 text-chili text-sm font-body">{error}</p>}
      </section>

      {parsed && (
        <div className="flex flex-wrap gap-2 justify-center mb-6">
          <span className="font-mono text-xs bg-clay-light text-turmeric px-3 py-1 rounded-full border border-turmeric/30">
            {parsed.is_city_wide ? "🌆 " : "📍 "}
            {parsed.locality}
          </span>
          {parsed.cuisines.map((c) => (
            <span key={c} className="font-mono text-xs bg-clay-light text-curry px-3 py-1 rounded-full border border-curry/30">
              🌿 {c}
            </span>
          ))}
        </div>
      )}

      {recommendations.length > 0 && (
        <section>
          <h2 className="font-display text-2xl text-cream mb-5 text-center">
            Fresh off the pass
          </h2>
          <div className="grid gap-6 sm:grid-cols-2">
            {recommendations.map((rec, i) => (
              <RecTicket key={`${rec.name}-${rec.location}`} rec={rec} index={i} />
            ))}
          </div>
        </section>
      )}
    </>
  );
}

export default FindFoodTab;
