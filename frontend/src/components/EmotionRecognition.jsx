import React, { useState, useCallback, useRef } from 'react'; // Importing React and hooks
import { Upload, AlertCircle, Play, Mic, Clock, HelpCircle, X, RefreshCcw, Home } from 'lucide-react'; // Icons
import { Alert as CustomAlert, AlertDescription, AlertTitle } from './ui/alert'; // Custom Alert component
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'; // For rendering charts
import './EmotionRecognition.css'; // Custom styles
import logo from './Logo.jpg'; // App logo
import Recorder from 'recorder-js';
//import WaveSurfer from 'wavesurfer.js';

export default function EmotionRecognition() {
  // State variables
  const [file, setFile] = useState(null); // For uploaded file
  const [audioUrl, setAudioUrl] = useState(null); // URL of the audio file
  const [recording, setRecording] = useState(false); // Recording state
  const [loading, setLoading] = useState(false); // Loading state
  const [results, setResults] = useState(null); // Emotion analysis results
  const [error, setError] = useState(null); // Error message
  const [showHelpModal, setShowHelpModal] = useState(false); // Help modal visibility
  const audioContextRef = useRef(null); // Reference for audio context
  const recorderRef = useRef(null); // Reference for recorder
  
  // Handle file drop/upload
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer?.files[0] || e.target.files[0];

    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      setFile(droppedFile);
      setAudioUrl(URL.createObjectURL(droppedFile)); // Set audio URL for playback
      setError(null);
      handleUpload(droppedFile); // Upload and analyze
    } else {
      setError('Please upload an audio file'); // Error for invalid file
    }
  }, []);

  // Handle file upload and emotion analysis
  const handleUpload = async (audio) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('audio', audio);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json(); // Parse response
      setResults(data); // Set results
    } catch (err) {
      setError(err.message); // Handle error
    } finally {
      setLoading(false);
    }
  };

  // Refresh/reset the application
  const handleRefresh = () => {
    setFile(null);
    setAudioUrl(null); // Clear the audio URL
    setResults(null);
    setError(null);
  };

  // Start recording audio
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const recorder = new Recorder(audioContext, { type: 'audio/wav' });

      recorder.init(stream);
      recorderRef.current = recorder;
      audioContextRef.current = audioContext;
      recorder.start();

      setRecording(true);
      setError(null);
    } catch (err) {
      setError('Microphone access denied or unavailable.');
    }
  }, []);

  // Stop recording audio
  const stopRecording = useCallback(async () => {
    if (recorderRef.current) {
      const recorder = recorderRef.current;

      const { blob } = await recorder.stop();
      setAudioUrl(URL.createObjectURL(blob)); // Set audio URL for playback
      setRecording(false);

      setLoading(true); // Indicate loading while analyzing emotions
      await handleUpload(blob); // Automatically upload and process the recording
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  }, []);

  // Prepare data for the chart
  const emotionData = results?.emotions ? 
    Object.entries(results.emotions)
      .map(([name, value]) => ({
        name,
        value: (value * 100).toFixed(1),
      }))
      .sort((a, b) => b.value - a.value) : [];

  // Get color based on emotion confidence
  const getEmotionColor = (value) => {
    value = parseFloat(value);
    if (value > 75) return '#4F46E5';  // Strong emotion
    if (value > 50) return '#818CF8';  // Moderate emotion
    return '#C7D2FE';  // Weak emotion
  };

  // Steps for help modal
  const helpSteps = [
    "Step 1: Upload your audio file by clicking 'Upload' button or by dragging the file.",
    "Step 2: You can choose to start a recording if you prefer live input instead of uploading.",
    "Step 3: Wait for the system to analyze your audio input or file.",
    "Step 4: View the emotion analysis and emotional dimensions on the results section.",
    "Step 5: The uploaded files or inputs activity will be stored in history for future view."
  ];

  // Toggle help modal visibility
  const toggleHelpModal = () => setShowHelpModal(!showHelpModal);

  return (
    <div className="container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="logo">
          <img src={logo} alt="Logo" className="logo-image" />
        </div>
        <ul className="nav-links">
          <li>
            <a href="/homepage">
              <Home className="nav-icon" />
              <span>Home</span>
            </a>
          </li>
          <li>
            <a href="/History">
              <Clock className="nav-icon" />
              <span>History</span>
            </a>
          </li>
          <li onClick={toggleHelpModal}>
            <HelpCircle className="nav-icon" />
            <span>Help</span>
          </li>
        </ul>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="content-wrapper">
          <h1 className="text-3xl font-bold text-center mb-8">
            Speech Emotion Recognition
          </h1>

          {/* Upload Area */}
          <div
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('file-input').click()}
          >
            <input
              id="file-input"
              type="file"
              className="hidden"
              accept="audio/*"
              onChange={handleDrop}
            />
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-2">Drop your audio file here or click to browse</p>
            {file && <p className="text-sm text-gray-500 mt-2">Selected: {file.name}</p>}
          </div>

          <br />

          {/* Audio Player */}
          {audioUrl && (
            <div className="mt-4 text-center">
              <audio controls src={audioUrl} className="mx-auto">
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
          <br />

          {/* Refresh Button */}
          <div className="RefreshButton">
            <button
              className="refresh-button absolute right-4 p-4 rounded-full bg-gray-500 text-white transition-colors hover:opacity-90 flex items-center justify-center"
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering file input
                handleRefresh();
              }}
              style={{ top: '50%', transform: 'translateY(-50%)' }}
            >
              <RefreshCcw className="h-6 w-6" />
            </button>
          </div>

          {/* Record Button */}
          <div className="text-center">
            <button
              className={`p-4 rounded-full ${recording ? 'bg-red-500' : 'bg-blue-500'} text-white transition-colors hover:opacity-90`}
              onClick={recording ? stopRecording : startRecording}
            >
              {recording ? <Play className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
            </button>
            {recording && <p className="text-sm text-gray-500 mt-2">Recording...</p>}
          </div>
          <br />

          {/* Error Message */}
          {error && (
            <CustomAlert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </CustomAlert>
          )}

          {/* Loading State */}
          {loading && (
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto" />
              <p className="mt-2">Analyzing audio...</p>
            </div>
          )}

          {/* Results */}
          {results && (
            <div className="space-y-6">
              {/* Emotions Chart */}
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Detected Emotions</h2>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={emotionData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={60} interval={0} />
                      <YAxis
                        label={{
                          value: 'Confidence (%)',
                          angle: -90,
                          position: 'insideLeft',
                        }}
                        domain={[0, 100]}
                      />
                      <Tooltip formatter={(value) => [`${value}%`, 'Confidence']} />
                      <Bar dataKey="value" fill="#4F46E5" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Emotional Dimensions */}
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Emotional Dimensions</h2>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(results.dimensions).map(([dim, value]) => (
                    <div key={dim} className="text-center">
                      <h3 className="font-medium capitalize">{dim}</h3>
                      <p className="text-2xl font-bold">
                        {(value * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-500">
                        {results.interpretation[dim]}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Help Modal */}
      {showHelpModal && (
        <div
          className="fixed inset-0 bg-gray-800 bg-opacity-75 flex justify-center items-center z-50"
          onClick={toggleHelpModal}
        >
          <div
            className="bg-white rounded-lg p-6 w-1/2 max-w-lg shadow-md"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-2xl font-semibold mb-4">Help</h2>
            <ol className="list-decimal ml-6 space-y-2">
              {helpSteps.map((step, index) => (
                <li key={index}>{step}</li>
              ))}
            </ol>
            <button
              className="mt-6 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={toggleHelpModal}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
