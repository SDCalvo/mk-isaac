-- Isaac Game State Reader (Fixed Pipe Version)
-- This mod communicates with a Python script using files

local mod = RegisterMod("IsaacGameStateReader", 1)

-- Turn on debugging
local DEBUG = true

-- Print a debug message to log
function DebugLog(message)
    if DEBUG then
        -- Try to log to Isaac's debug console
        Isaac.DebugString("[GameStateReader] " .. message)
        
        -- Also try to write to a debug log file
        local debugFile = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader/output/debug_log.txt"
        local file = io.open(debugFile, "a")
        if file then
            file:write(os.date("%Y-%m-%d %H:%M:%S") .. " - " .. message .. "\n")
            file:close()
        end
    end
end

DebugLog("Fixed Mod initialized with pipe communication")

-- Pipe paths
local pipePaths = {
    input = nil,
    output = nil
}

-- Try to read pipe info
function ReadPipeInfo()
    local pipeInfoPath = "E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader/output/pipe_info.txt"
    local file = io.open(pipeInfoPath, "r")
    if file then
        local content = file:read("*all")
        file:close()
        
        DebugLog("Read pipe info: " .. content)
        
        for line in content:gmatch("[^\r\n]+") do
            local key, value = line:match("([^=]+)=(.*)")
            if key and value then
                if key == "INPUT_PIPE" then
                    pipePaths.input = value
                    DebugLog("Found input pipe: " .. value)
                elseif key == "OUTPUT_PIPE" then
                    pipePaths.output = value
                    DebugLog("Found output pipe: " .. value)
                end
            end
        end
        
        return pipePaths.input and pipePaths.output
    else
        DebugLog("Failed to read pipe info")
        return false
    end
end

-- Write to the output pipe
function WriteToPipe(data)
    if not pipePaths.output then
        DebugLog("Cannot write to output pipe: pipe path not set")
        return false
    end
    
    local file = io.open(pipePaths.output, "w")
    if file then
        file:write(data)
        file:close()
        DebugLog("Wrote to output pipe: " .. string.sub(data, 1, 50) .. "...")
        return true
    else
        DebugLog("Failed to write to output pipe")
        return false
    end
end

-- Read from the input pipe
function ReadFromPipe()
    if not pipePaths.input then
        DebugLog("Cannot read from input pipe: pipe path not set")
        return nil
    end
    
    local file = io.open(pipePaths.input, "r")
    if file then
        local content = file:read("*all")
        file:close()
        
        if content and content ~= "" then
            DebugLog("Read from input pipe: " .. content)
            
            -- Clear the input pipe
            local clearFile = io.open(pipePaths.input, "w")
            if clearFile then
                clearFile:write("")
                clearFile:close()
            end
            
            return content
        else
            return nil
        end
    else
        DebugLog("Failed to read from input pipe")
        return nil
    end
end

-- Process commands
function ProcessCommand(command)
    if command == "status" then
        -- Create a status report using pcall to catch any errors
        local status, result = pcall(function()
            local game = Game()
            local player = Isaac.GetPlayer(0)
            local level = game:GetLevel()
            local room = game:GetRoom()
            
            -- Use simple string building for reliability
            local response = "Status Report:\n"
            response = response .. "Time: " .. os.date() .. "\n"
            response = response .. "Frame: " .. game:GetFrameCount() .. "\n"
            response = response .. "Player Health: " .. player:GetHearts() .. "/" .. player:GetMaxHearts() .. "\n"
            
            -- Add enemy information
            local enemies = {}
            for _, entity in ipairs(Isaac.GetRoomEntities()) do
                if entity:IsVulnerableEnemy() and not entity:IsDead() then
                    table.insert(enemies, {
                        type = entity.Type,
                        hp = entity.HitPoints,
                        position = {x = entity.Position.X, y = entity.Position.Y}
                    })
                end
            end
            
            response = response .. "Enemy Count: " .. #enemies .. "\n"
            for i, enemy in ipairs(enemies) do
                if i <= 3 then
                    response = response .. "Enemy " .. i .. ": Type=" .. enemy.type .. 
                               ", HP=" .. enemy.hp .. 
                               ", Pos=(" .. enemy.position.x .. "," .. enemy.position.y .. ")\n"
                end
            end
            
            response = response .. "Player Position: " .. player.Position.X .. ", " .. player.Position.Y .. "\n"
            response = response .. "Floor: " .. level:GetName() .. " (Stage " .. level:GetStage() .. ")\n"
            response = response .. "Room Type: " .. room:GetType() .. "\n"
            response = response .. "Room Clear: " .. tostring(room:IsClear()) .. "\n"
            
            return response
        end)
        
        if status then
            -- Success
            WriteToPipe(result)
        else
            -- Error occurred
            DebugLog("Error in creating status: " .. tostring(result))
            WriteToPipe("Error: " .. tostring(result))
        end
    elseif command == "move_up" then
        -- Simulate key press
        Input.SetKeyboardControl(Keyboard.KEY_W, true)
        WriteToPipe("Moving up")
    elseif command == "move_down" then
        Input.SetKeyboardControl(Keyboard.KEY_S, true)
        WriteToPipe("Moving down")
    elseif command == "move_left" then
        Input.SetKeyboardControl(Keyboard.KEY_A, true)
        WriteToPipe("Moving left")
    elseif command == "move_right" then
        Input.SetKeyboardControl(Keyboard.KEY_D, true)
        WriteToPipe("Moving right")
    elseif command == "use_item" then
        Input.SetKeyboardControl(Keyboard.KEY_SPACE, true)
        WriteToPipe("Using item")
    elseif command == "bomb" then
        Input.SetKeyboardControl(Keyboard.KEY_E, true)
        WriteToPipe("Placing bomb")
    elseif command == "shoot_up" then
        Input.SetKeyboardControl(Keyboard.KEY_UP, true)
        WriteToPipe("Shooting up")
    elseif command == "shoot_down" then
        Input.SetKeyboardControl(Keyboard.KEY_DOWN, true)
        WriteToPipe("Shooting down")
    elseif command == "shoot_left" then
        Input.SetKeyboardControl(Keyboard.KEY_LEFT, true)
        WriteToPipe("Shooting left")
    elseif command == "shoot_right" then
        Input.SetKeyboardControl(Keyboard.KEY_RIGHT, true)
        WriteToPipe("Shooting right")
    else
        WriteToPipe("Unknown command: " .. command)
    end
end

-- Update function
local frameCount = 0
local hasReadPipeInfo = false
function mod:OnUpdate()
    frameCount = frameCount + 1
    
    -- Try to read pipe info at the start
    if not hasReadPipeInfo and frameCount > 10 then
        hasReadPipeInfo = ReadPipeInfo()
        if hasReadPipeInfo then
            DebugLog("Successfully read pipe info")
            WriteToPipe("Mod initialized at " .. os.date())
        else
            DebugLog("Failed to read pipe info")
        end
    end
    
    -- Check for commands every 30 frames
    if hasReadPipeInfo and frameCount % 30 == 0 then
        local command = ReadFromPipe()
        if command then
            DebugLog("Processing command: " .. command)
            ProcessCommand(command)
        end
    end
    
    -- Release keys that were pressed via commands
    if frameCount % 3 == 0 then
        pcall(function()
            Input.SetKeyboardControl(Keyboard.KEY_W, false)
            Input.SetKeyboardControl(Keyboard.KEY_A, false)
            Input.SetKeyboardControl(Keyboard.KEY_S, false)
            Input.SetKeyboardControl(Keyboard.KEY_D, false)
            Input.SetKeyboardControl(Keyboard.KEY_SPACE, false)
            Input.SetKeyboardControl(Keyboard.KEY_E, false)
            Input.SetKeyboardControl(Keyboard.KEY_UP, false)
            Input.SetKeyboardControl(Keyboard.KEY_DOWN, false)
            Input.SetKeyboardControl(Keyboard.KEY_LEFT, false)
            Input.SetKeyboardControl(Keyboard.KEY_RIGHT, false)
        end)
    end
end

-- Register callbacks
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.OnUpdate)

-- Create a test file to verify file I/O is working
local testFile = io.open("E:/Steam/steamapps/common/The Binding of Isaac Rebirth/mods/IsaacGameStateReader/output/test.txt", "w")
if testFile then
    testFile:write("Fixed mod loaded at " .. os.date())
    testFile:close()
    DebugLog("Created test file")
else
    DebugLog("Failed to create test file")
end

return mod 