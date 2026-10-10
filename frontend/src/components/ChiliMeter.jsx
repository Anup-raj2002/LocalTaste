function ChiliMeter({ similarity }) {
  const filled = Math.max(1, Math.round(similarity * 5));
  return (
    <div className="flex gap-0.5 items-center" title={`${Math.round(similarity * 100)}% match`}>
      {[0, 1, 2, 3, 4].map((i) => (
        <span key={i} className={i < filled ? "opacity-100" : "opacity-15"} style={{ fontSize: "0.85rem" }}>
          🌶️
        </span>
      ))}
    </div>
  );
}

export default ChiliMeter;