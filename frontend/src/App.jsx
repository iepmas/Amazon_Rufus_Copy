export default function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold text-blue-600">Amazon Rufus Copy</h1>
      <p className="mt-4 text-lg text-gray-700">Multimodal shopping assistant</p>

      <div className="mt-10 w-full max-w-xl">
        {/* Text Search UI */}
        <form className="flex flex-col gap-4 bg-white p-6 rounded shadow">
          <label className="font-medium text-gray-800">Ask a question</label>
          <input
            type="text"
            className="border border-gray-300 rounded px-4 py-2"
            placeholder="e.g. What is a good laptop with 16GB RAM?"
          />
          <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
            Submit
          </button>
        </form>

        {/* Image Search UI */}
        <form
          className="flex flex-col gap-4 bg-white p-6 rounded shadow mt-8"
          encType="multipart/form-data"
        >
          <label className="font-medium text-gray-800">Upload an image</label>
          <input type="file" className="border border-gray-300 rounded px-4 py-2" />
          <button className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
            Search by Image
          </button>
        </form>
      </div>
    </div>
  );
}
