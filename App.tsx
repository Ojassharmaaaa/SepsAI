import { useState } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './components/Login';
import Register from './components/Register';
import Dashboard from './components/Dashboard';
import Landing from './pages/Landing';

type Page = 'landing' | 'login' | 'register' | 'dashboard';

function AppContent() {
  const { user, loading } = useAuth();
  const [currentPage, setCurrentPage] = useState<Page>('landing');
  const [showRegister, setShowRegister] = useState(false);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (user) {
  return <Dashboard onLogout={() => setCurrentPage('landing')} />;
}


  if (currentPage === 'login') {
    return (
      <Login
        onToggleRegister={() => {
          setShowRegister(true);
          setCurrentPage('register');
        }}
      />
    );
  }

  if (currentPage === 'register') {
    return (
      <Register
        onToggleLogin={() => {
          setShowRegister(false);
          setCurrentPage('login');
        }}
      />
    );
  }

  return (
    <Landing
      onGetStarted={() => setCurrentPage('login')}
      onNavigate={(page) => setCurrentPage(page as Page)}
    />
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
