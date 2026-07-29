import { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { SocketProvider } from './contexts/SocketContext';
import LoginPage from './pages/LoginPage';
import ChatPage from './pages/ChatPage';
import FilesPage from './pages/FilesPage';
import { requestNotificationPermission } from './utils/notifications';

type Tab = 'chat' | 'files';

function MainLayout() {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>('chat');

  useEffect(() => {
    requestNotificationPermission();
  }, []);

  if (!user) return null;

  return (
    <SocketProvider>
      <div className="app-viewport flex min-h-0 overflow-hidden">
        {/* Navigation Rail */}
        <div className="w-16 bg-gray-900 flex flex-col items-center py-4 gap-1 flex-shrink-0">
          {/* Logo */}
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white font-bold text-lg mb-4">
            H
          </div>

          {/* Chat Tab */}
          <button
            onClick={() => setActiveTab('chat')}
            className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
              activeTab === 'chat'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
            title="Messenger"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </button>

          {/* Files Tab */}
          <button
            onClick={() => setActiveTab('files')}
            className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
              activeTab === 'files'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
            title="File Manager"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
          </button>

          {/* Spacer */}
          <div className="flex-1" />

          {/* User & Logout */}
          <div className="flex flex-col items-center gap-2">
            <div className="w-9 h-9 bg-gray-700 rounded-full flex items-center justify-center text-white text-sm font-medium" title={user.name}>
              {user.name.charAt(0).toUpperCase()}
            </div>
            <button
              onClick={logout}
              className="w-10 h-10 rounded-xl text-gray-500 hover:text-white hover:bg-gray-800 flex items-center justify-center transition"
              title="Logout"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 min-h-0 h-full overflow-hidden">
          {activeTab === 'chat' && <ChatPage />}
          {activeTab === 'files' && <FilesPage />}
        </div>
      </div>
    </SocketProvider>
  );
}

function AppContent() {
  const { user } = useAuth();

  if (!user) {
    return <LoginPage />;
  }

  return <MainLayout />;
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
