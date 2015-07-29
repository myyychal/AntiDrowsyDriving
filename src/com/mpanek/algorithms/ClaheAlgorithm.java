package com.mpanek.algorithms;

import org.opencv.core.Mat;

public class ClaheAlgorithm {
	
	private int currentClipLimit = 4;
	private int currentTileSize = 4;

	public ClaheAlgorithm() {
		this.currentClipLimit = 4;
		this.currentTileSize = 4;
	}

	public ClaheAlgorithm(int currentClipLimit, int currentTileSize) {
		this.currentClipLimit = currentClipLimit;
		this.currentTileSize = currentTileSize;
	}
	
	public void process(Mat frame){
		ApplyCLAHEExt(frame.getNativeObjAddr(), (double) currentClipLimit,
				(double) currentTileSize, (double) currentTileSize);
	}
	
	public native void ApplyCLAHEExt(long matAddrSrc, double clipLimit,
			double tileWidth, double tileHeight);

	public int getCurrentClipLimit() {
		return currentClipLimit;
	}
	
	public void setCurrentClipLimit(int currentClipLimit) {
		this.currentClipLimit = currentClipLimit;
	}
	
	public int getCurrentTileSize() {
		return currentTileSize;
	}
	
	public void setCurrentTileSize(int currentTileSize) {
		this.currentTileSize = currentTileSize;
	}
	
}
