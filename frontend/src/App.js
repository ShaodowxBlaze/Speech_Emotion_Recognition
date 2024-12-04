import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import EmotionRecognition from './components/EmotionRecognition';
import History from './components/History'; // Import your History component
import { Home } from 'lucide-react';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<EmotionRecognition />} />
          <Route path="/homepage" element={<Home />} />
          <Route path="/history" element={<History />} />
          <Route path="/EmotionRecognition" element={<EmotionRecognition />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;