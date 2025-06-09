import React, { useState } from "react";

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

export default function App() {
  const [question, setQuestion] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [responseText, setResponseText] = useState("");
  const [imageResults, setImageResults] = useState([]);

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${BASE_URL}/api/text-search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setResponseText(data.response?.result || "No result.");
    } catch (err) {
      console.error("Text search failed:", err);
      setResponseText("Error occurred.");
    }
  };

  const handleImageSubmit = async (e) => {
    e.preventDefault();
    if (!imageFile) return;

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const res = await fetch(`${BASE_URL}/api/image-search`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setImageResults(data.results || []);
    } catch (err) {
      console.error("Image search failed:", err);
      setImageResults([]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold text-blue-600">Amazon Rufus Copy</h1>
      <p className="mt-4 text-lg text-gray-700">Multimodal shopping assistant</p>

      <div className="mt-10 w-full max-w-xl">

        {/* Text Search */}
        <form
          onSubmit={handleTextSubmit}
          className="flex flex-col gap-4 bg-white p-6 rounded shadow"
        >
          <label className="font-medium text-gray-800">Ask a question</label>
          <input
            type="text"
            className="border border-gray-300 rounded px-4 py-2"
            placeholder="e.g. Show me red dresses for summer"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            type="submit"
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Submit
          </button>
          {responseText && (
            <p className="mt-2 text-gray-800 bg-gray-100 p-2 rounded">{responseText}</p>
          )}
        </form>

        {/* Image Search */}
        <form
          onSubmit={handleImageSubmit}
          className="flex flex-col gap-4 bg-white p-6 rounded shadow mt-8"
          encType="multipart/form-data"
        >
          <label className="font-medium text-gray-800">Upload an image</label>
          <input
            type="file"
            className="border border-gray-300 rounded px-4 py-2"
            onChange={(e) => setImageFile(e.target.files[0])}
          />
          <button
            type="submit"
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          >
            Search by Image
          </button>

          {imageResults.length > 0 && (
            <div className="mt-4">
              <h2 className="text-lg font-semibold mb-2">Top Matches:</h2>
              <ul className="space-y-2">
                {imageResults.map(([score, item], idx) => (
                  <li key={idx} className="bg-gray-100 p-2 rounded text-sm">
                    <p><strong>Score:</strong> {score.toFixed(4)}</p>
                    <p><strong>Name:</strong> {item.productDisplayName}</p>
                    <p><strong>Category:</strong> {item.subCategory}</p>
                    <p><strong>Gender:</strong> {item.gender}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </form>

      </div>
    </div>
  );
}
