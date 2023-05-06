import math

#globals
numTilesBallPosX = 10
numTilesBallPosY = 10
numTilesPlayPos = 10
numTilesVelocityX = 10
numTilesVelocityY = 10 
numTilings = 4
numTiles = numTilings*numTilesBallPosX*numTilesBallPosY*numTilesPlayPos*numTilesVelocityX*numTilesVelocityY

# 0 <= ball_pos_x <= 82
# 0 <= ball_pos_y <= 72
# 0 <= player_pos <= 72
# -5 <= ball_vel_x <= 5
# -5 <= ball_vel_y <= 5
ball_pos_x_max = [0.0,82.0]
ball_pos_y_max = [0.0,72.0]
play_pos_max = [0.0,72.0]
ball_vel_max = [-30.0,30.0]


# tileSize = (max - min)/tilingSize
ball_pos_x_tile_mov_val = -((ball_pos_x_max[1]-ball_pos_x_max[0])/(numTilesBallPosX-1))/numTilings
ball_pos_y_tile_mov_val = -((ball_pos_y_max[1]-ball_pos_y_max[0])/(numTilesBallPosY-1))/numTilings
ball_vel_x_tile_mov_val = -((ball_vel_max[1]-ball_vel_max[0])/(numTilesVelocityX-1))/numTilings
ball_vel_y_tile_mov_val = -((ball_vel_max[1]-ball_vel_max[0])/(numTilesVelocityY-1))/numTilings
player_pos_mov_val = -((play_pos_max[1]-play_pos_max[0])/(numTilesPlayPos-1))/numTilings

def tilecode(ball_pos_x,ball_pos_y,play_pos,ball_vel_x,ball_vel_y,tileIndices):
    # print(numTiles)
    for i in range (numTilings):
        
        ballPosXMovementConstant = i * ball_pos_x_tile_mov_val
        ballPosYMovementConstant = i * ball_pos_y_tile_mov_val
        ballVelXMovementConstant = i * ball_vel_x_tile_mov_val
        ballVelYMovementConstant = i * ball_vel_y_tile_mov_val
        playPosMovementConstant = i * player_pos_mov_val
        
        b_pos_x_c = int((numTilesBallPosX-1) * (ball_pos_x - ballPosXMovementConstant)/(ball_pos_x_max[1]))
        b_pos_y_c = int((numTilesBallPosY-1) * (ball_pos_y - ballPosYMovementConstant)/(ball_pos_y_max[1]))
        b_vel_x_c = int((numTilesVelocityX-1) * (ball_vel_x - ballVelXMovementConstant)/(ball_vel_max[1]-ball_vel_max[0]))
        b_vel_y_c = int((numTilesVelocityY-1) * (ball_vel_y - ballVelYMovementConstant)/(ball_vel_max[1]-ball_vel_max[0]))
        play_pos_c = int((numTilesPlayPos-1) * (play_pos - playPosMovementConstant)/(play_pos_max[1]))

        tileIndices[i] = i * (numTilesBallPosX*numTilesBallPosY*numTilesPlayPos*numTilesVelocityX*numTilesVelocityY) + ((play_pos*numTilesBallPosX*numTilesBallPosY*numTilesVelocityX*numTilesVelocityY) +(ball_vel_y*numTilesBallPosX*numTilesBallPosY*numTilesVelocityX) + (ball_vel_x*numTilesBallPosX*numTilesBallPosY) + (ball_pos_y*numTilesBallPosX) + ball_pos_x)
        tileIndices[i] = int(tileIndices[i])
    
def printTileCoderIndices(x,y):
    tileIndices = [-1]*numTilings
    tilecode(x,y,tileIndices)
    print('Tile indices for input (',x,',',y,') are : ', tileIndices)