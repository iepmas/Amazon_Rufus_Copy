/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",  // Ensures Tailwind is applied to all your component files
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
