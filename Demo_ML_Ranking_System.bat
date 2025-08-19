@echo off
title HAT ML Ranking System - Demo
echo.
echo ===============================================
echo    🚀 HAT ML RANKING SYSTEM - DEMO 🚀
echo ===============================================
echo.
echo This demo showcases the complete ML ranking system:
echo.
echo ✅ 10 trained ML models for hotel roles
echo ✅ Advanced semantic embeddings 
echo ✅ 10-feature candidate assessment
echo ✅ Bias detection and mitigation
echo ✅ Real-time ranking API
echo.
echo ===============================================
echo            DEMO OPTIONS
echo ===============================================
echo.
echo [1] Test ML ranking with sample resumes
echo [2] Train new models (if needed)
echo [3] Run system health check
echo [4] Launch web interface
echo [5] View implementation details
echo [6] Exit
echo.
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto test_ranking
if "%choice%"=="2" goto train_models
if "%choice%"=="3" goto health_check
if "%choice%"=="4" goto web_interface
if "%choice%"=="5" goto view_details
if "%choice%"=="6" goto exit

:test_ranking
echo.
echo ===============================================
echo         TESTING ML RANKING SYSTEM
echo ===============================================
echo.
echo Testing with Front Desk Agent position...
echo.
python -c "
import asyncio
import sys
import os
sys.path.append('.')

async def test_ml_ranking():
    try:
        from app.services.ranking_service import RankingService
        from pathlib import Path
        
        print('🔄 Initializing ML ranking service...')
        ranking_service = RankingService()
        
        # Sample job description
        job_description = '''
        Position: Front Desk Agent
        Department: Front Office
        
        We are seeking a friendly and professional Front Desk Agent to join our hotel team. 
        The ideal candidate will have excellent customer service skills, attention to detail, 
        and experience with hotel property management systems.
        
        Key Responsibilities:
        - Check guests in and out
        - Handle reservations and room assignments
        - Provide exceptional customer service
        - Process payments and maintain accurate records
        - Assist guests with inquiries and requests
        
        Requirements:
        - High school diploma or equivalent
        - 1-2 years customer service experience preferred
        - Strong communication skills
        - Proficiency with computer systems
        - Ability to work flexible hours including weekends
        '''
        
        # Check for sample resumes
        input_dir = Path('input_resumes')
        resume_files = list(input_dir.glob('*.docx')) + list(input_dir.glob('*.txt'))
        
        if not resume_files:
            print('⚠️  No resume files found in input_resumes folder')
            print('   Please add some resume files to test the system')
            return
            
        print(f'📄 Found {len(resume_files)} resume files to rank')
        print()
        
        # Rank candidates
        print('🤖 Running ML ranking...')
        results = await ranking_service.rank_candidates(
            job_description=job_description,
            role_id='front_desk_agent',
            resume_files=resume_files[:5]  # Test with first 5 resumes
        )
        
        print()
        print('===============================================')
        print('              RANKING RESULTS')
        print('===============================================')
        
        for i, result in enumerate(results[:3], 1):
            print(f'🏆 Rank #{i}: {result.get(\"filename\", \"Unknown\")}')
            print(f'   📊 ML Score: {result.get(\"ml_score\", 0):.3f}')
            print(f'   🎯 Semantic Similarity: {result.get(\"semantic_similarity\", 0):.3f}')
            print(f'   💼 Experience Score: {result.get(\"experience_score\", 0):.3f}')
            print(f'   🎓 Education Score: {result.get(\"education_score\", 0):.3f}')
            print()
            
        print('✅ ML ranking test completed successfully!')
        
    except Exception as e:
        print(f'❌ Error during ML ranking test: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_ml_ranking())
"
echo.
pause
goto menu

:train_models
echo.
echo ===============================================
echo         TRAINING ML MODELS
echo ===============================================
echo.
echo Training models for all hotel roles...
echo.
python scripts\train_ml_models.py --all
echo.
pause
goto menu

:health_check
echo.
echo ===============================================
echo         SYSTEM HEALTH CHECK
echo ===============================================
echo.
python scripts\test_ml_system.py
echo.
pause
goto menu

:web_interface
echo.
echo ===============================================
echo         LAUNCHING WEB INTERFACE
echo ===============================================
echo.
echo Starting Streamlit web application...
echo Open your browser to: http://localhost:8501
echo.
streamlit run enhanced_streamlit_app.py
pause
goto menu

:view_details
echo.
echo ===============================================
echo       IMPLEMENTATION DETAILS
echo ===============================================
echo.
type ML_RANKING_IMPLEMENTATION_COMPLETE.md
echo.
pause
goto menu

:menu
cls
echo.
echo ===============================================
echo    🚀 HAT ML RANKING SYSTEM - DEMO 🚀
echo ===============================================
echo.
echo This demo showcases the complete ML ranking system:
echo.
echo ✅ 10 trained ML models for hotel roles
echo ✅ Advanced semantic embeddings 
echo ✅ 10-feature candidate assessment
echo ✅ Bias detection and mitigation
echo ✅ Real-time ranking API
echo.
echo ===============================================
echo            DEMO OPTIONS
echo ===============================================
echo.
echo [1] Test ML ranking with sample resumes
echo [2] Train new models (if needed)
echo [3] Run system health check
echo [4] Launch web interface
echo [5] View implementation details
echo [6] Exit
echo.
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto test_ranking
if "%choice%"=="2" goto train_models
if "%choice%"=="3" goto health_check
if "%choice%"=="4" goto web_interface
if "%choice%"=="5" goto view_details
if "%choice%"=="6" goto exit

:exit
echo.
echo ===============================================
echo Thank you for testing the HAT ML Ranking System!
echo ===============================================
echo.
echo The system is ready for production use with:
echo ✅ Complete ML infrastructure
echo ✅ Trained models for 10 hotel roles  
echo ✅ Advanced bias safety measures
echo ✅ Comprehensive API endpoints
echo.
echo For support: See TROUBLESHOOTING_GUIDE.md
echo.
pause
exit
