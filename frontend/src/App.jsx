import React, { useState } from 'react'
import SearchBar from './components/SearchBar'
import SimilarQuestionsTable from './components/SimilarQuestionsTable'
import AlertDialog from './components/AlertDialog'
import SimilarQ_logo from './assets/file.png'
import {FaGithub} from 'react-icons/fa'

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [alert, setAlert] = useState({
    open:false,
    message:""
  });
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(0);
  const [selectedDifficulties, setSelectedDifficulties]= useState([])
  const [selectedTags, setSelectedTags]= useState([])
  const [showTagDropdown, setShowTagDropdown] = useState(false);
  const [showDifficultyDropdown, setShowDifficultyDropdown] = useState(false);
  const [tagSearch, setTagSearch] = useState("");

  const allTags=Array.from(
    new Set(results.flatMap(q=>q.tags))
  ).sort();

  const filteredResults=results.filter(q=>
    (selectedDifficulties.length===0 || selectedDifficulties.includes(q.difficulty)) &&
    (selectedTags.length===0|| q.tags.some(tag=>selectedTags.includes(tag)))
  ).sort((a,b)=>b.score-a.score);

  const handleSearch= async (input)=>{
    if(!input.trim()) return;
    setLoading(true);
    setAlert({open:false, message:""});

    const verifyRes=await fetch('https://similarq.onrender.com/api/verify-question',{
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({query:input})
    })
    const verifyData=await verifyRes.json()
    if(!verifyData.valid){
      setAlert({open:true,message:"Question Not Found: Check the name you have entered or try entering the Problem url.\
        If these dont work then we might not have this question in our Database."})
      setResults([])
      setLoading(false)
      return
    }

    const simResponse=await fetch('https://similarq.onrender.com/api/similar_search',{
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({query:verifyData.metadata})
    })
    // console.log(simResponse)
    const data=await simResponse.json()
    // console.log(data)
    if(!data || data['results'].length===0){
      setAlert({open:true,message:"No similar questions found."})
      setResults([])
    }else{
      setResults(data['results'])
      setPage(0)
      setSelectedDifficulties([])
      setSelectedTags([])
    }
    setLoading(false)
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-lcblack">
      <div className={`transition-all duration-700 ${results.length ? "mt-10" : "mt-40"} w-full flex flex-col items-center`}>
        <div className='flex items-center justifyc-center'>
          <img src={SimilarQ_logo} width="100" height="100"></img>
          <h1 className="font-extrabold mb-4">
            <span className='text-lcwhite drop-shadow text-6xl ml-2'>Similar</span>
            <span className='text-lcgold drop-shadow ml-2 text-7xl'>Q</span>
          </h1>
        </div>
        <div className="mb-1 text-lcgray2 text-xl font-medium">
          Enter the question name or LeetCode URL to find similar questions. 
        </div>
        <SearchBar
          value={query}
          setValue={setQuery}
          onSearch={handleSearch}
          loading={loading}
        />
      </div>
      <AlertDialog open={alert.open} message={alert.message} onClose={() => setAlert({ open: false, message: "" })} />
      {results.length > 0 && ( 
        <SimilarQuestionsTable
          results={filteredResults}
          page={page}
          setPage={setPage}
          selectedDifficulties={selectedDifficulties}
          setSelectedDifficulties={setSelectedDifficulties}
          selectedTags={selectedTags}
          setSelectedTags={setSelectedTags}
          allTags={allTags}
          showTagDropdown={showTagDropdown}
          setShowTagDropdown={setShowTagDropdown}
          showDifficultyDropdown={showDifficultyDropdown}
          setShowDifficultyDropdown={setShowDifficultyDropdown}
          tagSearch={tagSearch}
          setTagSearch={setTagSearch}
        />
      )}
      <footer className="w-full bg-lcgray py-3 mt-auto flex justify-evenly items-center px-2">

        <a
          href="https://github.com/Sid-Purk/SimilarQ"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaGithub size={24} color='white'/>
        </a>
        <div className="text-lcwhite text-md text-center max-w-full px-4">
          <span className="font-bold">Can't find a question?</span> Try entering the question URL in the format:<span> </span>
          <code className='bg-black rounded-2xl p-1'>https://leetcode.com/problems/problem-name/</code><br/>
          If that doesn't work, the question might not be present in our database.
          Please <a target="_blank" rel="noopener noreferrer" href="https://github.com/Sid-Purk/SimilarQ/issues" className="text-lcblue underline">create an issue</a> in the repo.
        </div>
        <div className='text-lcwhite'>
          You can find the Solution in<span> </span>
          <a
            href="https://github.com/kamyu104/LeetCode-Solutions"
            target="_blank"
            rel="noopener noreferrer"
            className="text-lcblue underline font-medium"
          >
            this Repo <i class="fas fa-external-link-alt"></i>
          </a>
        </div>
      </footer>
    </div>
  );
}

export default App
