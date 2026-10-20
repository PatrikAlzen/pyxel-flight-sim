from flightsim.app import main

# Terrain workers are separate processes that re-import the main module, so
# the game must only start when this file is run directly.
if __name__ == "__main__":
    main()
