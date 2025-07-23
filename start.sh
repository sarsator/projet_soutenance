#!/bin/bash
# Script de démarrage rapide pour Gaia Vision
# Usage: ./start.sh
echo "DÉMARRAGE RAPIDE GAIA VISION"

# Aller dans le dossier du projet
cd /home/sarsator/projets/gaia_vision

# Vérifier l'environnement virtuel
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Environnement virtuel non trouvé"
    echo "Créez-le avec: python -m venv .venv"
    exit 1
fi

echo "✅ Environnement virtuel trouvé"

# Nettoyer les processus existants
echo "🧹 Nettoyage..."
pkill -f 'python.*api' 2>/dev/null
pkill -f 'python.*frontend' 2>/dev/null
pkill -f 'tensorboard' 2>/dev/null
lsof -ti:8000,5000,6006 | xargs -r kill -9 2>/dev/null

# Lancer les services
echo "Lancement des services..."

# API
echo "API..."
.venv/bin/python api/main.py &
API_PID=$!

# Frontend  
echo "Frontend..."
.venv/bin/python frontend/app.py &
FRONTEND_PID=$!

# TensorBoard (optionnel)
echo "   TensorBoard..."
if [ -f "tensorboard.sh" ]; then
    bash tensorboard.sh &
    TENSORBOARD_PID=$!
else
    echo "tensorboard.sh non trouvé"
fi

# Attendre
echo "⏳ Attente du démarrage..."
sleep 10

echo ""
echo " SERVICES DISPONIBLES:"
echo "    Frontend:    http://localhost:5000"
echo "    API:         http://localhost:8000"
echo "    TensorBoard: http://localhost:6006"
echo "    API Docs:    http://localhost:8000/docs"
echo ""
echo "Pour arrêter: python stop.py"
echo "   ou: pkill -f 'python.*api'; pkill -f 'python.*frontend'; pkill -f 'tensorboard'"
echo ""
echo "✅ Tous les services sont lancés !"

# Attendre Ctrl+C
trap 'echo ""; echo "🛑 Arrêt des services..."; kill $API_PID $FRONTEND_PID $TENSORBOARD_PID 2>/dev/null; exit 0' INT
echo "Appuyez sur Ctrl+C pour arrêter tous les services..."
wait
