import React, {useEffect, useRef} from "react";
const PAGE_SIZE=10;

const difficultyColors = {
  Easy: "text-lcneongreen",
  Medium: "text-lcgold",
  Hard: "text-lcredDark",
};

export default function SimilarQuestionsTable({ 
    results, 
    page, 
    setPage, 
    selectedDifficulties,
    setSelectedDifficulties,
    selectedTags,
    setSelectedTags,
    allTags,
    showTagDropdown,
    setShowTagDropdown,
    showDifficultyDropdown,
    setShowDifficultyDropdown,
    tagSearch,
    setTagSearch
}){
  const start = page * PAGE_SIZE;
  const end = start + PAGE_SIZE;
  const pageResults = results.slice(start, end);

    // Difficulty dropdown
  const difficulties = ["Easy", "Medium", "Hard"];
  const handleDifficultyChange = (diff) => {
    setSelectedDifficulties((prev) =>
      prev.includes(diff)
        ? prev.filter((d) => d !== diff)
        : [...prev, diff]
    );
  };

  // Tag dropdown
  const handleTagChange = (tag) => {
    setSelectedTags((prev) =>
      prev.includes(tag)
        ? prev.filter((t) => t !== tag)
        : [...prev, tag]
    );
  };

  const tagDropdownRef=useRef()
  const difDropdownRef=useRef()

  useEffect(()=>{
    function handleClickOutside(event){
      if(showTagDropdown&&tagDropdownRef.current&&!tagDropdownRef.current.contains(event.target)){
        setShowTagDropdown(false)
      }
    }
    document.addEventListener("mousedown",handleClickOutside);
    return ()=>{
      document.removeEventListener("mousedown",handleClickOutside)
    }
  },[showTagDropdown]);

  useEffect(()=>{
    function handleClickOutside(event){
      if(showDifficultyDropdown&&difDropdownRef.current&&!difDropdownRef.current.contains(event.target)){
        setShowDifficultyDropdown(false)
      }
    }
    document.addEventListener("mousedown",handleClickOutside);
    return ()=>{
      document.removeEventListener("mousedown",handleClickOutside)
    }
  },[showDifficultyDropdown])

  return (
    <div className="mt-10 w-full max-w-6xl">
        <table className="w-full rounded-xl shadow-lg">
          <thead>
            <tr className="bg-lcgray text-lcwhite">
              <th className="py-3 px-2 font-semibold text-center">Sl.No</th>
              <th className="py-3 px-2 font-semibold text-center">Question Name</th>
              <th className="py-3 px-2 font-semibold text-center">Similarity Score</th>
              <th className="py-3 px-2 font-semibold text-center relative">
                  <button
                      className="flex items-center justify-center w-full bg-transparent text-lcwhite font-semibold"
                      onClick={() => setShowTagDropdown((v) => !v)}
                  >
                      <span className="px-1">Tags</span> <i class="fas fa-filter"></i>
                  </button>
                  {showTagDropdown && (
                      <div ref={tagDropdownRef} className="absolute left-1/2 transform -translate-x-1/2 mt-2 w-56 bg-lcdark rounded-xl shadow-lg z-50 border border-lcgray">
                          <input
                              type="text"
                              value={tagSearch}
                              onChange={(e) => setTagSearch(e.target.value)}
                              placeholder="Search tags..."
                              className="w-full px-3 py-2 bg-lcblack text-lcwhite rounded-t-xl border-b border-lcgray focus:outline-none"
                          />
                          <div className="max-h-60 overflow-y-auto">
                              {allTags
                              .filter((tag) => tag.toLowerCase().includes(tagSearch.toLowerCase()))
                              .map((tag) => (
                                  <label
                                  key={tag}
                                  className={`flex items-center px-3 py-1 cursor-pointer text-lcwhite hover:bg-lcgray ${
                                      selectedTags.includes(tag) ? "bg-lcgray2 text-lcblack font-bold" : ""
                                  }`}
                                  >
                                  <input
                                      type="checkbox"
                                      checked={selectedTags.includes(tag)}
                                      onChange={() => handleTagChange(tag)}
                                      className="mr-2 accent-lcblue"
                                  />
                                  <span>{tag}</span>
                                  </label>
                              ))}
                          </div>
                      </div>
                  )}
              </th>
              <th className="py-3 px-2 font-semibold text-center relative">
                  <button
                      className="flex items-center justify-center w-full bg-transparent text-lcwhite font-semibold"
                      onClick={() => setShowDifficultyDropdown((v) => !v)}
                  >
                      <span className="px-1">Difficulty</span> <i className="px-1" class="fas fa-filter"></i>
                  </button>
                  {showDifficultyDropdown && (
                      <div ref={difDropdownRef} className="absolute left-1/2 transform -translate-x-1/2 mt-2 w-40 bg-lcdark rounded-xl shadow-lg z-50 border border-lcgray">
                      {difficulties.map((diff) => (
                          <label
                          key={diff}
                          className={`flex items-center px-3 py-1 cursor-pointer text-lcwhite hover:bg-lcgray ${
                              selectedDifficulties.includes(diff) ? "bg-lcgray2 text-lcblack font-bold" : ""
                          }`}
                          >
                          <input
                              type="checkbox"
                              checked={selectedDifficulties.includes(diff)}
                              onChange={() => handleDifficultyChange(diff)}
                              className="mr-2 accent-lcgold"
                          />
                          <span>{diff}</span>
                          </label>
                      ))}
                      </div>
                  )}
              </th>
              <th className="py-3 px-2 font-semibold text-center">Acceptance Rate</th>
            </tr>
          </thead>
          <tbody>
            {pageResults.map((q, idx) => (
              <tr key={q.name} className={
                  idx % 2 === 0
                    ? "bg-lcdark"
                    : "bg-lcblack"
              }>
                <td className="py-3 text-center text-lcwhite">{start + idx + 1}</td>
                <td className="py-3 text-center">
                  <a
                    href={q.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-lcwhite hover:text-lcblue font-semibold"
                  >
                    {q.name}
                  </a>
                  {q.isPaid && (
                    <span
                      title="Paid Only"
                      className="ml-2 text-lcgold"
                      role="img"
                      aria-label="lock"
                    >
                      &#x1F512;
                    </span>
                  )}
                </td>
                <td className="py-3 text-center text-lcwhite">{(q.score *100).toFixed(2)}</td>
                <td className="py-3 text-center">
                  <div className="flex flex-wrap gap-2 justify-center">
                    {q.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-3 py-1 rounded-full bg-lcgray2 text-lcblack text-xs font-medium shadow-sm"
                      //   style={{
                      //     background:
                      //       "linear-gradient(90deg, #f7b267 0%, #ffd580 100%)",
                      //     color: "#333",
                      //   }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </td>
                <td className={`py-3 text-center rounded-lg font-semibold ${difficultyColors[q.difficulty]}`}>
                  {q.difficulty}
                </td>
                <td className="py-3 text-center text-lcwhite">{q.acRate ? `${q.acRate}` : "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        
              
      <div className="flex justify-center items-center gap-4 m-4">
        <button
          onClick={() => setPage(page - 1)}
          disabled={page === 0}
          className="px-4 py-2 rounded bg-lcgray2 text-lcblack hover:bg-lcgray disabled:opacity-50 shadow cursor-pointer"
        >
          Previous
        </button>
        <span className="font-semibold text-lcwhite">
          Page {page + 1} of {Math.ceil(results.length / PAGE_SIZE)}
        </span>
        <button
          onClick={() => setPage(page + 1)}
          disabled={end >= results.length}
          className="px-4 py-2 rounded bg-lcgray2 text-lcblack hover:bg-lcgray disabled:opacity-50 shadow cursor-pointer"
        >
          Next
        </button>
      </div>
    </div>
  );
}