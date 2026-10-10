import ChiliMeter from "./ChiliMeter";

function RecTicket({ rec, index }) {
  const tilt = index % 2 === 0 ? "-1deg" : "1deg";
  return (
    <div
      className="chit-enter torn-top bg-cream text-ink rounded-b-md px-5 pb-5 shadow-lg"
      style={{ "--tilt": tilt, transform: `rotate(${tilt})`, animationDelay: `${index * 80}ms` }}
    >
      <div className="flex items-start justify-between gap-3 border-b border-ink/15 pb-2 mb-2">
        <h3 className="font-display text-xl font-semibold leading-tight">{rec.name}</h3>
        <ChiliMeter similarity={rec.similarity} />
      </div>

      <p className="font-mono text-[0.7rem] tracking-wide uppercase text-ink/60 mb-2">
        {rec.cuisines}
      </p>

      <div className="flex justify-between font-mono text-xs text-ink/70 mb-3 border-b border-dashed border-ink/20 pb-2">
        <span>RATING</span>
        <span>{rec.rate ?? "N/A"} ★</span>
      </div>

      {rec.top_dishes.length > 0 && (
        <div className="mb-3">
          <p className="font-mono text-[0.65rem] uppercase tracking-widest text-ink/50 mb-1">Order this</p>
          <ul className="space-y-0.5">
            {rec.top_dishes.map((dish) => (
              <li key={dish} className="flex items-center gap-2 font-mono text-xs text-ink/80">
                <span className="flex-1 truncate">{dish}</span>
                <span className="flex-1 border-b border-dotted border-ink/25" />
              </li>
            ))}
          </ul>
        </div>
      )}

      {rec.explanation && (
        <p className="font-display italic text-sm text-chili/90 leading-snug mt-2">
          "{rec.explanation}"
        </p>
      )}
    </div>
  );
}

export default RecTicket;