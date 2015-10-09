
-- misc utilities

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

-- Multi byte characters start with a byte with bits 7 and 8 set, trailing bytes have bit 7 not set and bit 8 set.
-- https://forums.coronalabs.com/topic/42019-split-utf-8-string-word-with-foreign-characters-to-letters/ by ingemar
function UTF8ToCharArray(str)
    local charArray = {};
    local iStart = 0;
    local strLen = str:len();

    local function bit(b)
        return 2 ^ (b - 1);
    end

    local function hasbit(w, b)
        return w % (b + b) >= b;
    end

    local checkMultiByte = function(i)
        if (iStart ~= 0) then
            charArray[#charArray + 1] = str:sub(iStart, i - 1);
            iStart = 0;
        end
    end

    for i = 1, strLen do
        local b = str:byte(i);
        local multiStart = hasbit(b, bit(7)) and hasbit(b, bit(8));
        local multiTrail = not hasbit(b, bit(7)) and hasbit(b, bit(8));

        if (multiStart) then
            checkMultiByte(i);
            iStart = i;

        elseif (not multiTrail) then
            checkMultiByte(i);
            charArray[#charArray + 1] = str:sub(i, i);
        end
    end

    -- process if last character is multi-byte
    checkMultiByte(strLen + 1);

    return charArray;
end
