-- FVP's seek bar (mpv's own can't be used: the film comes through a pipe, with no length and no seeking).
-- A strip along the bottom: where the film is / how long it is (hh:mm:ss), a bar to click or drag, the
-- volume. Shown while the mouse moves, during a seek, and when the player asks. The player (fvp.py) tells
-- it the film's length and does the seeking: "script-message sv seekto <seconds>".

local assdraw = require 'mp.assdraw'

local ov = mp.create_osd_overlay("ass-events")
local duration = 0          -- the film's length, s (told by the player)
local target, target_until = nil, 0   -- where a seek is going (shown until the picture gets there)
local visible_until = 0
local dragging, drag_pos = false, nil
local bound = false
local timer = nil

local function clock(t)
    t = math.max(0, math.floor(t + 0.5))
    return string.format("%d:%02d:%02d", math.floor(t / 3600), math.floor(t % 3600 / 60), t % 60)
end

-- the strip: x0..x1 is the bar, y0 the strip's top
local function geometry()
    local w, h = mp.get_osd_size()
    if not w or w <= 0 then return nil end
    local bh = math.max(34, h * 0.06)
    local fs = bh * 0.50
    local side = fs * 6.2          -- room for "0:00:00 / 0:00:00" left of the bar, the volume right of it
    return {w = w, h = h, bh = bh, fs = fs, x0 = fs * 0.8 + side * 1.75, x1 = w - fs * 0.8 - side * 0.75,
            y0 = h - bh}
end

local function pos_at(g, x)
    local f = (x - g.x0) / math.max(1, g.x1 - g.x0)
    return math.min(1, math.max(0, f)) * duration
end

local function now_pos()
    if dragging and drag_pos then return drag_pos end
    if target and mp.get_time() < target_until then return target end
    return mp.get_property_number("time-pos", 0) or 0
end

local function draw()
    local g = geometry()
    if not g or (mp.get_time() > visible_until and not dragging) then
        ov.data = ""
        ov:update()
        if timer then timer:stop() end
        return
    end
    ov.res_x, ov.res_y = g.w, g.h
    local pos = now_pos()
    local f = duration > 0 and math.min(1, math.max(0, pos / duration)) or 0
    local a = assdraw.ass_new()
    local mid = g.y0 + g.bh / 2
    local th = math.max(4, g.bh * 0.16)       -- the bar's thickness
    local function box(x0, y0, x1, y1, colour, alpha)
        a:new_event()
        a:pos(0, 0)
        a:append(string.format("{\\an7\\bord0\\shad0\\1c&H%s&\\1a&H%s&}", colour, alpha))
        a:draw_start()
        a:rect_cw(x0, y0, x1, y1)
        a:draw_stop()
    end
    box(0, g.y0, g.w, g.h, "000000", "50")                                  -- the strip
    box(g.x0, mid - th / 2, g.x1, mid + th / 2, "FFFFFF", "A0")             -- the bar, empty
    local x = g.x0 + (g.x1 - g.x0) * f
    box(g.x0, mid - th / 2, x, mid + th / 2, "FFFFFF", "00")                -- filled so far
    box(x - th * 0.6, mid - th * 1.6, x + th * 0.6, mid + th * 1.6, "FFFFFF", "00")  -- the knob
    local function text(an, px, s)
        a:new_event()
        a:pos(px, mid)
        a:append(string.format("{\\an%d\\fs%d\\bord%.1f\\shad0\\1c&HFFFFFF&\\3c&H000000&\\b1}", an, g.fs, g.fs * 0.06))
        a:append(s)
    end
    text(4, g.fs * 0.8, clock(pos) .. " / " .. clock(duration))
    local vol = mp.get_property_number("volume", 100) or 100
    text(6, g.w - g.fs * 0.8, mp.get_property_bool("mute") and "muted" or string.format("vol %d %%", vol))
    ov.data = a.text
    ov:update()
end

local function show(secs)
    visible_until = math.max(visible_until, mp.get_time() + (secs or 2.5))
    if not timer then
        timer = mp.add_periodic_timer(0.2, draw)
    elseif not timer:is_enabled() then
        timer:resume()
    end
    draw()
end

local function click(e)
    local g = geometry()
    if not g or duration <= 0 then return end
    local m = mp.get_property_native("mouse-pos") or {}
    if e.event == "down" then
        dragging, drag_pos = true, pos_at(g, m.x or 0)
        show(3)
    elseif e.event == "up" and dragging then
        dragging = false
        if drag_pos then
            target, target_until = drag_pos, mp.get_time() + 4
            mp.commandv("script-message", "sv", "seekto", string.format("%.2f", drag_pos))
        end
        show(3)
    end
end

-- the left button is taken only while the mouse is on the strip (elsewhere it stays mpv's: double click
-- for full screen, dragging the window)
local function on_mouse(_, m)
    local g = geometry()
    if not g or not m then return end
    local over = m.hover and m.y >= g.y0 and mp.get_time() <= visible_until + 0.3
    if dragging then
        drag_pos = pos_at(g, m.x)
        show(3)
    elseif m.hover then
        show(over and 3 or 2.5)
    end
    if (over or dragging) and not bound then
        mp.add_forced_key_binding("MBTN_LEFT", "fvp-bar", click, {complex = true})
        bound = true
    elseif not over and not dragging and bound then
        mp.remove_key_binding("fvp-bar")
        bound = false
    end
end

mp.observe_property("mouse-pos", "native", on_mouse)
mp.observe_property("volume", "number", function() show(2) end)
mp.observe_property("mute", "bool", function() show(2) end)
mp.observe_property("pause", "bool", function() show(2.5) end)
mp.observe_property("osd-dimensions", "native", function() if mp.get_time() <= visible_until then draw() end end)

mp.register_script_message("fvp-ui", function(what, value)
    if what == "duration" then
        duration = tonumber(value) or 0
    elseif what == "seeking" then
        target, target_until = tonumber(value), mp.get_time() + 4
        show(3)
    elseif what == "arrived" then
        target = nil
    elseif what == "show" then
        show(tonumber(value) or 3)
    end
end)
