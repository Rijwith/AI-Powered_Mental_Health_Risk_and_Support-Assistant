import React, { useState } from "react";

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello 👋 I'm your Mental Health Assistant. How are you feeling today?" }
  ]);
  const [input, setInput] = useState("");
  const [showDashboard, setShowDashboard] = useState(false);
  const [dashboardType, setDashboardType] = useState("present"); // "present" or "overall"

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { sender: "user", text: input }];
    setMessages(newMessages);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input })
      });

      const data = await res.json();
      setMessages([...newMessages, { sender: "bot", text: data.reply || "..." }]);
    } catch (error) {
      setMessages([...newMessages, { sender: "bot", text: "⚠️ Server error. Try again later." }]);
    }

    setInput("");
  };

  const toggleDashboardType = () => {
    setDashboardType(dashboardType === "present" ? "overall" : "present");
  };

  // Determine iframe URL
  const dashboardURL =
    dashboardType === "present"
      ? "http://127.0.0.1:8000/reports/present"
      : "http://127.0.0.1:8000/reports/overall";

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      {/* Chat container */}
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-lg flex flex-col p-4 mb-4">
        <div className="flex-1 overflow-y-auto mb-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`my-2 p-3 rounded-xl max-w-xs ${
                msg.sender === "user"
                  ? "ml-auto bg-blue-500 text-white"
                  : "mr-auto bg-gray-200 text-black"
              }`}
            >
              {msg.text}
            </div>
          ))}
        </div>

        <div className="flex mb-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            className="flex-1 border rounded-xl p-2 mr-2"
            placeholder="Type your message..."
          />
          <button
            onClick={sendMessage}
            className="bg-blue-500 text-white px-4 py-2 rounded-xl hover:bg-blue-600"
          >
            Send
          </button>
        </div>

        {/* Dashboard buttons */}
        <div className="flex gap-2 flex-wrap mb-2">
          <button
            onClick={() => setShowDashboard(!showDashboard)}
            className="bg-green-500 text-white px-4 py-2 rounded-xl hover:bg-green-600"
          >
            {showDashboard ? "Hide Dashboard" : "View Dashboard"}
          </button>

          {showDashboard && (
            <button
              onClick={toggleDashboardType}
              className="bg-yellow-500 text-white px-4 py-2 rounded-xl hover:bg-yellow-600"
            >
              {dashboardType === "present"
                ? "View Overall Sessions Dashboard"
                : "View Present Session Dashboard"}
            </button>
          )}
        </div>
      </div>

      {/* Dashboard iframe */}
      {showDashboard && (
        <div className="w-full max-w-5xl h-[600px] bg-white rounded-2xl shadow-lg p-2 overflow-auto">
          <h2 className="text-center font-bold mb-2">
            {dashboardType === "present"
              ? `Present Session Dashboard`
              : `Overall Sessions Dashboard`}
          </h2>
          <iframe
            key={dashboardType} // forces re-render when dashboard type changes
            src={dashboardURL}
            title="Chatbot Dashboard"
            width="100%"
            height="100%"
            className="rounded-xl border"
          />
        </div>
      )}
    </div>
  );
}

export default App;
