import React from 'react';
import { Upload, Clock } from 'lucide-react';
import './EmotionRecognition.css';
import logo from './Logo.jpg';

export default function History() {
  // Example data
  const historyData = [
    { no: 1, fileName: 'audio1.wav', duration: '2:35', date: '2024-11-03', time: '14:23', status: 'Success', results: 'Happy, Calm' },
    { no: 2, fileName: 'audio2.wav', duration: '3:12', date: '2024-11-03', time: '15:10', status: 'Failed', results: '' },
    // Add more rows as needed
  ];

  return (
    <div className="container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="logo">
          <img src={logo} alt="Logo" className="logo-image" />
        </div>
        <ul className="nav-links">
          <li>
            <a href="/History">
              <Clock className="nav-icon" />
              <span>History</span>
            </a>
          </li>
          <li>
            <a href="/EmotionRecognition">
              <Upload className="nav-icon" />
              <span>Upload</span>
            </a>
          </li>
        </ul>
      </div>

      {/* Main Content */}
      <div className="history-container">
        <h2 className="text-3xl font-bold text-center mb-8">Upload History</h2>

        <table className="w-full border-collapse border border-gray-200">
          <thead>
            <tr className="bg-gray-100">
              <th className="border border-gray-300 px-4 py-2">No</th>
              <th className="border border-gray-300 px-4 py-2">File Name</th>
              <th className="border border-gray-300 px-4 py-2">Duration</th>
              <th className="border border-gray-300 px-4 py-2">Date</th>
              <th className="border border-gray-300 px-4 py-2">Time</th>
              <th className="border border-gray-300 px-4 py-2">Upload Status</th>
              <th className="border border-gray-300 px-4 py-2">Results</th>
            </tr>
          </thead>
          <tbody>
            {historyData.map((entry) => (
              <tr key={entry.no} className="text-center">
                <td className="border border-gray-300 px-4 py-2">{entry.no}</td>
                <td className="border border-gray-300 px-4 py-2">{entry.fileName}</td>
                <td className="border border-gray-300 px-4 py-2">{entry.duration}</td>
                <td className="border border-gray-300 px-4 py-2">{entry.date}</td>
                <td className="border border-gray-300 px-4 py-2">{entry.time}</td>
                <td className={`border border-gray-300 px-4 py-2 ${entry.status === 'Success' ? 'text-green-600' : 'text-red-600'}`}>{entry.status}</td>
                <td className="border border-gray-300 px-4 py-2">{entry.results || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
