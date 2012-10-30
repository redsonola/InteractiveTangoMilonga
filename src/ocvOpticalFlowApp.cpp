#include "cinder/app/AppBasic.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
#include "cinder/qtime/QuickTime.h"
#include "cinder/params/Params.h"
#include "cinder/Surface.h"
#include "cinder/gl/Texture.h"
#include "cinder/Text.h"
#include "cinder/Utilities.h"
#include "cinder/ImageIo.h"

#include "CinderOpenCV.h"

using namespace ci;
using namespace ci::app;

class ocvOpticalFlowApp : public AppBasic {
  public:
	void setup();
	void update();
	void draw();

	void keyDown( KeyEvent event );
	void chooseFeatures( cv::Mat currentFrame );
	void trackFeatures( cv::Mat currentFrame );
	void loadMovieFile( const fs::path &path ); 
    void findPeople( cv::Mat curImg ); 
	
	gl::Texture                     mTexture;
	Capture                         mCapture;
	cv::Mat                         mPrevFrame;
    std::vector<cv::Point2f>		mPrevFeatures, mFeatures;
    std::vector<uint8_t>			mFeatureStatuses;
	bool                            mDrawPoints;
    
	//for detecting people
    cv::HOGDescriptor hog;
	cv::Mat mono_img;
    std::vector<cv::Rect> mFoundPeople;

    
	gl::Texture					mFrameTexture, mInfoTexture;
	qtime::MovieSurface         mMovie;
	
	static const int MAX_FEATURES = 300;
};

void ocvOpticalFlowApp::setup()
{
	mDrawPoints = true;
	
//	mCapture = Capture( 640, 480 );
//	mCapture.start();
    
	fs::path moviePath = getOpenFilePath();
	if( ! moviePath.empty() )
		loadMovieFile( moviePath );  

    hog.setSVMDetector( cv::HOGDescriptor::getDefaultPeopleDetector() );

}


void ocvOpticalFlowApp::findPeople(cv::Mat curImg)
{
    cv::Mat monoImg; 
    cv::cvtColor(curImg, monoImg, CV_BGR2GRAY);
    hog.detectMultiScale(monoImg, mFoundPeople);
}


void ocvOpticalFlowApp::keyDown( KeyEvent event )
{
	if( event.getChar() == 'p' ) {
		mDrawPoints = ! mDrawPoints;
	}
	else if( event.getChar() == 'u' ) {
		chooseFeatures( mPrevFrame );
	}
	else if( event.getChar() == 'o' ) {
		fs::path moviePath = getOpenFilePath();
		if( ! moviePath.empty() )
			loadMovieFile( moviePath );
	}    
    
}

void ocvOpticalFlowApp::loadMovieFile( const fs::path &moviePath )
{
	try {
		// load up the movie, set it to loop, and begin playing
    	mMovie = qtime::MovieSurface( moviePath );    
		mMovie.setLoop();
		mMovie.play();
		
		// create a texture for showing some info about the movie
		TextLayout infoText;
		infoText.clear( ColorA( 0.2f, 0.2f, 0.2f, 0.5f ) );
		infoText.setColor( Color::white() );
		infoText.addCenteredLine( moviePath.filename().string() );
		infoText.addLine( toString( mMovie.getWidth() ) + " x " + toString( mMovie.getHeight() ) + " pixels" );
		infoText.addLine( toString( mMovie.getDuration() ) + " seconds" );
		infoText.addLine( toString( mMovie.getNumFrames() ) + " frames" );
		infoText.addLine( toString( mMovie.getFramerate() ) + " fps" );
		infoText.setBorder( 4, 2 );
		mInfoTexture = gl::Texture( infoText.render( true ) );
	}
	catch( ... ) {
		console() << "Unable to load the movie." << std::endl;
		mMovie.reset();
		mInfoTexture.reset();
	}
    
	mFrameTexture.reset();
}


void ocvOpticalFlowApp::chooseFeatures( cv::Mat currentFrame )
{
	cv::goodFeaturesToTrack( currentFrame, mFeatures, MAX_FEATURES, 0.005, 3.0 );
}

void ocvOpticalFlowApp::trackFeatures( cv::Mat currentFrame )
{
    std::vector<float> errors;
	mPrevFeatures = mFeatures;
	cv::calcOpticalFlowPyrLK( mPrevFrame, currentFrame, mPrevFeatures, mFeatures, mFeatureStatuses, errors );
}

void ocvOpticalFlowApp::update()
{ 
    if( !mMovie ) return; 
    if (!mMovie.getSurface() ) return; 

	Surface surface = mMovie.getSurface();
    cv::Mat currentFrame( toOcv( Channel( surface ) ) );
    mTexture = gl::Texture( surface );
//    if( mPrevFrame.data ) 
//    {
//			if( mFeatures.empty() || getElapsedFrames() % 30 == 0 ) // pick new features once every 30 frames, or the first frame
//				chooseFeatures( mPrevFrame );
//			trackFeatures( currentFrame );
//    }
//    mPrevFrame = currentFrame;
    
    findPeople(currentFrame);
    
//	if( mCapture.checkNewFrame() ) {
//		Surface surface( mCapture.getSurface() );
//		mTexture = gl::Texture( surface );
//		cv::Mat currentFrame( toOcv( Channel( surface ) ) );
//		if( mPrevFrame.data ) {
//			if( mFeatures.empty() || getElapsedFrames() % 30 == 0 ) // pick new features once every 30 frames, or the first frame
//				chooseFeatures( mPrevFrame );
//			trackFeatures( currentFrame );
//		}
//		mPrevFrame = currentFrame;
//	}
    
}

void ocvOpticalFlowApp::draw()
{
	if( ( ! mTexture ) || mPrevFeatures.empty() )
		return;

	gl::clear();
	gl::enableAlphaBlending();
	
	gl::setMatricesWindow( getWindowSize() );
	glColor3f( 1, 1, 1 );
	gl::draw( mTexture, getWindowBounds() );
	
	glDisable( GL_TEXTURE_2D );
	glColor4f( 1, 1, 0, 0.5f );
	
//	if( mDrawPoints ) {
//		// draw all the old points
//		for( std::vector<cv::Point2f>::const_iterator featureIt = mPrevFeatures.begin(); featureIt != mPrevFeatures.end(); ++featureIt )
//			gl::drawStrokedCircle( fromOcv( *featureIt ), 4 );
//
//		// draw all the new points
//		for( std::vector<cv::Point2f>::const_iterator featureIt = mFeatures.begin(); featureIt != mFeatures.end(); ++featureIt )
//			gl::drawSolidCircle( fromOcv( *featureIt ), 4 );
//	}
	
	// draw the lines connecting them
//	glColor4f( 0, 1, 0, 0.5f );
//	glBegin( GL_LINES );
//	for( size_t idx = 0; idx < mFeatures.size(); ++idx ) {
//		if( mFeatureStatuses[idx] ) {
//			gl::vertex( fromOcv( mFeatures[idx] ) );
//			gl::vertex( fromOcv( mPrevFeatures[idx] ) );
//		}
//	}
//	glEnd();
    
    for(int i=0; i<mFoundPeople.size(); i++ )
    {
        cv::Rect r = mFoundPeople[i];
        Rectf faceRect( fromOcv( r ) );
        gl::drawStrokedRect( faceRect );        
    }
    
}


CINDER_APP_BASIC( ocvOpticalFlowApp, RendererGl )
