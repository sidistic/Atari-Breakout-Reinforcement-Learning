import math

#globals
numTiles = 4 * 100 * 10
numTilings = 4

# 0 <= ball_pos <= 82*72
# 0 <= player_pos <= 72
tilingSize = [(100)-1,9] # (subset)
# tileSize = (max - min)/tilingSize
ball_pos_tile_mov_val = -(82.0*72.0/tilingSize[0])/numTilings
player_pos_mov_val = -(72.0/tilingSize[1])/numTilings

# x = position, y = velocity
def tilecode(x,y,tileIndices):
    
    for i in range (numTilings):
        
        ballPosMovementConstant = i * ball_pos_tile_mov_val
        playPosMovementConstant = i * player_pos_mov_val
        
        xcoord = int(tilingSize[0] * (x - ballPosMovementConstant)/(82.0*72.0))
        ycoord = int(tilingSize[1] * (y - playPosMovementConstant)/(72.0))

        tileIndices[i] = i * (100*10) + ( ycoord * 100 + xcoord)
    # print(tileIndices)
    
    
def printTileCoderIndices(x,y):
    tileIndices = [-1]*numTilings
    tilecode(x,y,tileIndices)
    print('Tile indices for input (',x,',',y,') are : ', tileIndices)