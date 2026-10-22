import { useState } from "react";
import FindFoodTab from "./components/FindFoodTab";
import TasteTestTab from "./components/TasteTestTab";

function App() {
  const [activeTab, setActiveTab] = useState("find");

  return (
    <div className="min-h-screen text-cream font-body">
      <div className="max-w-2xl mx-auto px-5 py-12">
        <header className="mb-8 text-center">
          <h1 className="font-display text-5xl font-bold tracking-tight text-turmeric">
            LocalTaste
          </h1>
          <p className="text-cream/60 mt-2 text-sm font-mono uppercase tracking-widest">
            Order chit &middot; Bangalore
          </p>
        </header>

        <div className="flex justify-center gap-2 mb-8">
          <button
            onClick={() => setActiveTab("find")}
            className={`font-mono text-xs uppercase tracking-widest px-5 py-2.5 rounded-full border transition ${
              activeTab === "find"
                ? "bg-chili border-chili text-cream"
                : "bg-transparent border-cream/20 text-cream/50 hover:border-cream/40"
            }`}
          >
            🍽 Find Food
          </button>
          <button
            onClick={() => setActiveTab("feedback")}
            className={`font-mono text-xs uppercase tracking-widest px-5 py-2.5 rounded-full border transition ${
              activeTab === "feedback"
                ? "bg-turmeric border-turmeric text-ink"
                : "bg-transparent border-cream/20 text-cream/50 hover:border-cream/40"
            }`}
          >
            📝 Feedback
          </button>
        </div>

        {activeTab === "find" ? <FindFoodTab /> : <TasteTestTab />}
      </div>
    </div>
  );
}

export default App;
