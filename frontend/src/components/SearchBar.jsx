import React from "react";

export default function SearchBar({ value, setValue, onSearch, loading }){
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && value.trim()) {
      onSearch(value);
    }
  };
  return (
    <div className="flex gap-2 justify-center w-full">
      <input
        type="text"
        placeholder="Question name or LeetCode URL"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
        className="px-6 py-4 rounded-xl border-none bg-lcdark w-1/2 text-lg text-lcwhite placeholder-lcgray2 focus:outline-none focus:ring-2 focus:ring-lcblue shadow"
      />
      <button
        onClick={() => value.trim() && onSearch(value)}
        disabled={loading}
        className="px-8 py-4 rounded-xl bg-lcneongreen text-lcblack font-bold text-lg hover:bg-lcgreen hover:text-lcwhite transition shadow cursor-pointer"
      >
        Search
      </button>
    </div>
  );
}