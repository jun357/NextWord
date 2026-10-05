using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

[JsonConverter(typeof(StringEnumConverter))]
public enum EscapeAction
{
    NoAction,

    LookAround,

    InspectClock,
    InspectClockBack,

    ReadNote,

    InspectBookshelf,
    InspectBook,

    OpenSafe,
    TakeKey,

    OpenDrawer,

    InspectRadio,
    TurnOnRadio,

    InsertCassette,
    PlayCassette,
    Listen,

    InspectDoor,
    OpenDoor,

    Wait
}

public static class EscapeActionExtension
{
    public const EscapeAction NoAction = EscapeAction.NoAction;
}