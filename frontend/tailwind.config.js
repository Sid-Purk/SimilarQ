d.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,jsx,ts,tsx}", // if you have any JavaScript files
    ],
    safelist: [
      'bg-green-700', 'text-green-300', 'bg-neutral-900',
      'rounded-lg', 'my-4', 'p-6', 'm-3', 'p-2',
      'font-extrabold', 'text-5xl', 'font-sans', 'text-green-400',
      'pb-2', 'text-base', 'm-1', 'p-1', 'text-sm',
      'bg-neutral-800', 'placeholder-green-500', 'text-black',
      'bg-green-500', 'rounded-md', 'hover:text-white',
      'hover:bg-green-700', 'm-4', 'h-10', 'mx-4',
      'border-green-500', 'hover:bg-neutral-700', 'text-xs',
      'w-screen', 'h-12', 'border-t-green-500', 'p-2',
      'hover:text-green-500', 'transition-colors', 'hover:text-green-400'
    ],
    theme: {
        extend: {},
    },
    plugins: [],
}